use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader, Write};
use std::sync::mpsc::Sender;
use std::sync::OnceLock;
use std::fs::OpenOptions;
use serde_json::Value;

// Reuse StreamMessage and CancelToken from claude module
use crate::services::claude::{StreamMessage, CancelToken};

/// Cached path to the codex binary.
static CODEX_PATH: OnceLock<Option<String>> = OnceLock::new();

/// Resolve the path to the codex binary.
fn resolve_codex_path() -> Option<String> {
    if let Ok(output) = Command::new("which").arg("codex").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
    }

    // Fallback: use login shell to resolve PATH
    if let Ok(output) = Command::new("bash")
        .args(["-lc", "which codex"])
        .output()
    {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
    }

    None
}

/// Get the cached codex binary path.
fn get_codex_path() -> Option<&'static str> {
    CODEX_PATH.get_or_init(|| resolve_codex_path()).as_deref()
}

/// Debug logging helper (only active when COKACDIR_DEBUG=1)
fn debug_log(msg: &str) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = ENABLED.get_or_init(|| {
        std::env::var("COKACDIR_DEBUG").map(|v| v == "1").unwrap_or(false)
    });
    if !*enabled { return; }
    if let Some(home) = dirs::home_dir() {
        let debug_dir = home.join(".cokacdir").join("debug");
        let _ = std::fs::create_dir_all(&debug_dir);
        let log_path = debug_dir.join("codex.log");
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
        {
            let timestamp = chrono::Local::now().format("%H:%M:%S%.3f");
            let _ = writeln!(file, "[{}] {}", timestamp, msg);
        }
    }
}

/// Check if Codex CLI is available
pub fn is_codex_available() -> bool {
    #[cfg(not(unix))]
    { false }

    #[cfg(unix)]
    { get_codex_path().is_some() }
}

/// Execute a command using Codex CLI with streaming output.
///
/// Maps Codex CLI's JSON event stream to the same `StreamMessage` enum
/// used by `claude.rs`, so `telegram.rs` doesn't need to know which
/// backend is active.
pub fn execute_command_streaming(
    prompt: &str,
    session_id: Option<&str>,
    working_dir: &str,
    sender: Sender<StreamMessage>,
    system_prompt: Option<&str>,
    model: Option<&str>,
    cancel_token: Option<std::sync::Arc<CancelToken>>,
) -> Result<(), String> {
    debug_log("========================================");
    debug_log("=== codex execute_command_streaming START ===");
    debug_log("========================================");
    debug_log(&format!("prompt_len: {} chars", prompt.len()));
    debug_log(&format!("session_id: {:?}", session_id));
    debug_log(&format!("working_dir: {}", working_dir));

    let codex_bin = get_codex_path()
        .ok_or_else(|| {
            debug_log("ERROR: Codex CLI not found");
            "Codex CLI not found. Install with: npm install -g @openai/codex".to_string()
        })?;

    // Ensure working directory is a git repo (Codex CLI requires it)
    let work_dir_path = std::path::Path::new(working_dir);
    if !work_dir_path.join(".git").exists() {
        let _ = Command::new("git")
            .args(["init"])
            .current_dir(working_dir)
            .output();
        debug_log(&format!("Initialized git repo in {}", working_dir));
    }

    // Build args for: codex exec --json --full-auto [--model <m>] <prompt>
    let mut args: Vec<String> = vec![
        "exec".to_string(),
        "--json".to_string(),
        "--full-auto".to_string(),
    ];

    // Model selection
    if let Some(m) = model {
        args.push("--model".to_string());
        args.push(m.to_string());
    }

    // Resume session if available
    if let Some(sid) = session_id {
        // codex exec resume <session_id> "prompt"
        args[0] = "exec".to_string();
        args.push("resume".to_string());
        args.push(sid.to_string());
    }

    // Build the final prompt with system instructions prepended
    let final_prompt = if let Some(sp) = system_prompt {
        if sp.is_empty() {
            prompt.to_string()
        } else {
            format!("[System Instructions]\n{}\n\n[User Message]\n{}", sp, prompt)
        }
    } else {
        prompt.to_string()
    };

    args.push(final_prompt);

    debug_log("--- Spawning codex process ---");
    debug_log(&format!("Command: {} {}", codex_bin, args.join(" ")));

    let spawn_start = std::time::Instant::now();
    let mut cmd = Command::new(codex_bin);
    cmd.args(&args)
        .current_dir(working_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Explicitly pass OPENAI_API_KEY from environment
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        cmd.env("OPENAI_API_KEY", &api_key);
        debug_log(&format!("OPENAI_API_KEY set ({}... chars)", api_key.len()));
    } else {
        debug_log("WARNING: OPENAI_API_KEY not set!");
    }

    let mut child = cmd.spawn()
        .map_err(|e| {
            debug_log(&format!("ERROR: Failed to spawn: {}", e));
            format!("Failed to start Codex: {}. Is Codex CLI installed?", e)
        })?;
    debug_log(&format!("Codex process spawned in {:?}, pid={:?}", spawn_start.elapsed(), child.id()));

    // Store child PID in cancel token
    if let Some(ref token) = cancel_token {
        *token.child_pid.lock().unwrap() = Some(child.id());
    }

    // Read stdout line by line (JSON Lines stream)
    let stdout = child.stdout.take()
        .ok_or_else(|| "Failed to capture stdout".to_string())?;
    let reader = BufReader::new(stdout);

    let mut last_thread_id: Option<String> = None;
    let mut final_result: Option<String> = None;
    let mut line_count = 0;

    for line in reader.lines() {
        // Check cancel token
        if let Some(ref token) = cancel_token {
            if token.cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                debug_log("Cancel detected — killing codex process");
                let _ = child.kill();
                let _ = child.wait();
                return Ok(());
            }
        }

        let line = match line {
            Ok(l) => l,
            Err(e) => {
                debug_log(&format!("ERROR: Failed to read line: {}", e));
                let _ = sender.send(StreamMessage::Error {
                    message: format!("Failed to read output: {}", e)
                });
                break;
            }
        };

        line_count += 1;
        if line.trim().is_empty() { continue; }

        debug_log(&format!("Line {}: {}", line_count, &line.chars().take(200).collect::<String>()));

        if let Ok(json) = serde_json::from_str::<Value>(&line) {
            if let Some(msg) = parse_codex_event(&json, &mut last_thread_id, &mut final_result) {
                debug_log(&format!("  Parsed: {:?}", std::mem::discriminant(&msg)));
                if sender.send(msg).is_err() {
                    debug_log("  ERROR: Channel send failed");
                    break;
                }
            }
        }
    }

    debug_log(&format!("Total lines: {}, final_result: {}", line_count, final_result.is_some()));

    // Check cancel after loop
    if let Some(ref token) = cancel_token {
        if token.cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(());
        }
    }

    // Read stderr for error details
    let stderr_output = child.stderr.take()
        .map(|stderr| {
            let reader = BufReader::new(stderr);
            reader.lines()
                .filter_map(|l| l.ok())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();

    // Wait for process
    let status = child.wait().map_err(|e| format!("Process error: {}", e))?;

    // Send Done if not already sent
    if final_result.is_none() {
        let _ = sender.send(StreamMessage::Done {
            result: String::new(),
            session_id: last_thread_id.clone(),
        });
    }

    if !status.success() {
        let err_msg = if stderr_output.is_empty() {
            format!("Process exited with code {:?}", status.code())
        } else {
            format!("Codex error: {}", stderr_output.chars().take(500).collect::<String>())
        };
        debug_log(&format!("Process failed: {}", err_msg));
        return Err(err_msg);
    }

    debug_log("=== codex execute_command_streaming END (success) ===");
    Ok(())
}

/// Parse a Codex CLI JSON event into a StreamMessage.
///
/// Codex event types:
///   thread.started   → Init
///   item.started     (command_execution) → ToolUse
///   item.completed   (command_execution) → ToolResult
///   item.completed   (agent_message)     → Text
///   item.completed   (file_change)       → ToolResult (file edit info)
///   turn.completed   → (ignored, usage info)
///   error            → Error
///
/// Final agent_message text is also captured as the Done result.
fn parse_codex_event(
    json: &Value,
    last_thread_id: &mut Option<String>,
    final_result: &mut Option<String>,
) -> Option<StreamMessage> {
    let event_type = json.get("type")?.as_str()?;

    match event_type {
        "thread.started" => {
            let thread_id = json.get("thread_id")
                .and_then(|v| v.as_str())
                .unwrap_or("codex-session")
                .to_string();
            *last_thread_id = Some(thread_id.clone());
            Some(StreamMessage::Init { session_id: thread_id })
        }

        "item.started" => {
            let item = json.get("item")?;
            let item_type = item.get("type")?.as_str()?;

            match item_type {
                "command_execution" => {
                    let command = item.get("command")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    Some(StreamMessage::ToolUse {
                        name: "Bash".to_string(),
                        input: format!("{{\"command\": \"{}\"}}", command.replace('"', "\\\"")),
                    })
                }
                "web_search" => {
                    let query = item.get("query")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    Some(StreamMessage::ToolUse {
                        name: "WebSearch".to_string(),
                        input: format!("{{\"query\": \"{}\"}}", query.replace('"', "\\\"")),
                    })
                }
                "file_change" => {
                    let file = item.get("file")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let action = item.get("action")
                        .and_then(|v| v.as_str())
                        .unwrap_or("edit");
                    Some(StreamMessage::ToolUse {
                        name: "Edit".to_string(),
                        input: format!("{{\"file\": \"{}\", \"action\": \"{}\"}}", file, action),
                    })
                }
                _ => None,
            }
        }

        "item.completed" => {
            let item = json.get("item")?;
            let item_type = item.get("type")?.as_str()?;

            match item_type {
                "agent_message" => {
                    let text = item.get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if !text.is_empty() {
                        *final_result = Some(text.clone());
                        // Send as Done with the full message
                        Some(StreamMessage::Done {
                            result: text,
                            session_id: last_thread_id.clone(),
                        })
                    } else {
                        None
                    }
                }
                "command_execution" => {
                    let output = item.get("output")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let exit_code = item.get("exit_code")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    let is_error = exit_code != 0;
                    let content = if output.len() > 500 {
                        format!("{}...\n(truncated, exit_code={})", &output[..500], exit_code)
                    } else if output.is_empty() {
                        format!("(exit_code={})", exit_code)
                    } else {
                        format!("{}\n(exit_code={})", output, exit_code)
                    };
                    Some(StreamMessage::ToolResult { content, is_error })
                }
                "file_change" => {
                    let file = item.get("file")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let action = item.get("action")
                        .and_then(|v| v.as_str())
                        .unwrap_or("edit");
                    Some(StreamMessage::ToolResult {
                        content: format!("{}: {}", action, file),
                        is_error: false,
                    })
                }
                "reasoning" => {
                    // Reasoning items can be shown as text
                    let text = item.get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if !text.is_empty() {
                        Some(StreamMessage::Text { content: format!("💭 {}", text) })
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }

        "turn.completed" => {
            // Usage info — we can log but don't need to send as a message
            debug_log(&format!("Turn completed: {:?}", json.get("usage")));
            None
        }

        "turn.failed" => {
            let error = json.get("error")
                .and_then(|v| v.as_str())
                .or_else(|| json.get("message").and_then(|v| v.as_str()))
                .unwrap_or("Unknown error")
                .to_string();
            Some(StreamMessage::Error { message: error })
        }

        "error" => {
            let error = json.get("message")
                .and_then(|v| v.as_str())
                .or_else(|| json.get("error").and_then(|v| v.as_str()))
                .unwrap_or("Unknown error")
                .to_string();
            Some(StreamMessage::Error { message: error })
        }

        _ => {
            debug_log(&format!("Unknown event type: {}", event_type));
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_thread_started() {
        let json: Value = serde_json::from_str(
            r#"{"type":"thread.started","thread_id":"abc-123"}"#
        ).unwrap();
        let mut tid = None;
        let mut fr = None;

        match parse_codex_event(&json, &mut tid, &mut fr) {
            Some(StreamMessage::Init { session_id }) => {
                assert_eq!(session_id, "abc-123");
                assert_eq!(tid, Some("abc-123".to_string()));
            }
            _ => panic!("Expected Init"),
        }
    }

    #[test]
    fn test_parse_command_started() {
        let json: Value = serde_json::from_str(
            r#"{"type":"item.started","item":{"id":"i1","type":"command_execution","command":"ls -la","status":"in_progress"}}"#
        ).unwrap();
        let mut tid = None;
        let mut fr = None;

        match parse_codex_event(&json, &mut tid, &mut fr) {
            Some(StreamMessage::ToolUse { name, input }) => {
                assert_eq!(name, "Bash");
                assert!(input.contains("ls -la"));
            }
            _ => panic!("Expected ToolUse"),
        }
    }

    #[test]
    fn test_parse_command_completed() {
        let json: Value = serde_json::from_str(
            r#"{"type":"item.completed","item":{"id":"i1","type":"command_execution","command":"ls","output":"file.txt\n","exit_code":0}}"#
        ).unwrap();
        let mut tid = None;
        let mut fr = None;

        match parse_codex_event(&json, &mut tid, &mut fr) {
            Some(StreamMessage::ToolResult { content, is_error }) => {
                assert!(content.contains("file.txt"));
                assert!(!is_error);
            }
            _ => panic!("Expected ToolResult"),
        }
    }

    #[test]
    fn test_parse_agent_message() {
        let json: Value = serde_json::from_str(
            r#"{"type":"item.completed","item":{"id":"i3","type":"agent_message","text":"Done! The file has been created."}}"#
        ).unwrap();
        let mut tid = Some("thread-1".to_string());
        let mut fr = None;

        match parse_codex_event(&json, &mut tid, &mut fr) {
            Some(StreamMessage::Done { result, session_id }) => {
                assert_eq!(result, "Done! The file has been created.");
                assert_eq!(session_id, Some("thread-1".to_string()));
                assert_eq!(fr, Some("Done! The file has been created.".to_string()));
            }
            _ => panic!("Expected Done"),
        }
    }

    #[test]
    fn test_parse_error() {
        let json: Value = serde_json::from_str(
            r#"{"type":"error","message":"Rate limit exceeded"}"#
        ).unwrap();
        let mut tid = None;
        let mut fr = None;

        match parse_codex_event(&json, &mut tid, &mut fr) {
            Some(StreamMessage::Error { message }) => {
                assert_eq!(message, "Rate limit exceeded");
            }
            _ => panic!("Expected Error"),
        }
    }

    #[test]
    fn test_parse_unknown_type() {
        let json: Value = serde_json::from_str(
            r#"{"type":"unknown_event","data":"something"}"#
        ).unwrap();
        let mut tid = None;
        let mut fr = None;

        assert!(parse_codex_event(&json, &mut tid, &mut fr).is_none());
    }
}
