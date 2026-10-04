use clap::Args;

#[derive(Args, Debug)]
pub struct RadbArgs {
    /// Command and arguments to execute via radb (e.g., `shell`, `exec <cmd>`, `push`, `pull`)
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub args: Vec<String>,
}

pub fn handle_radb(args: RadbArgs) {
    let sub_args = args.args;

    if sub_args.is_empty() {
        let client = match radb::RuriClient::auto_connect() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[-] Could not connect to local wireless ADB: {}", e);
                eprintln!("[-] Ensure Wireless Debugging is enabled in Developer Options.");
                std::process::exit(1);
            }
        };
        println!(
            "[*] Connecting interactive shell to 127.0.0.1:{} (UID 2000)...",
            client.port()
        );
        if let Err(e) = client.open_shell() {
            eprintln!("[-] Shell error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    match sub_args[0].as_str() {
        "connect" => {
            let port: u16 = if sub_args.len() >= 2 {
                sub_args[1].parse().unwrap_or_else(|_| {
                    eprintln!("[-] Invalid port: {}", sub_args[1]);
                    std::process::exit(1);
                })
            } else {
                radb::scan_local_adbd(30000, 45000).unwrap_or_else(|| {
                    eprintln!("[-] Could not find active wireless ADB port on localhost.");
                    std::process::exit(1);
                })
            };

            let client = match radb::RuriClient::with_port(port) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("[-] Connection error: {}", e);
                    std::process::exit(1);
                }
            };
            println!(
                "[*] Connecting interactive shell to 127.0.0.1:{} (UID 2000)...",
                port
            );
            if let Err(e) = client.open_shell() {
                eprintln!("[-] Shell error: {}", e);
                std::process::exit(1);
            }
            return;
        }
        "pair" => {
            let (port, code) = if sub_args.len() >= 3 {
                (sub_args[1].clone(), sub_args[2].clone())
            } else {
                println!("[*] Opening Developer Options so you can tap 'Pair device with pairing code'...");
                let _ = std::process::Command::new("am")
                    .args([
                        "start",
                        "-a",
                        "android.settings.APPLICATION_DEVELOPMENT_SETTINGS",
                    ])
                    .output();

                use std::io::{stdin, stdout, Write};
                let mut p = String::new();
                let mut c = String::new();

                print!("[?] Enter pairing port shown on screen: ");
                let _ = stdout().flush();
                stdin().read_line(&mut p).expect("Failed to read port");

                print!("[?] Enter 6-digit pairing code: ");
                let _ = stdout().flush();
                stdin().read_line(&mut c).expect("Failed to read code");

                (p.trim().to_string(), c.trim().to_string())
            };

            let target = if port.contains(':') {
                port
            } else {
                format!("127.0.0.1:{}", port)
            };

            println!("[*] Pairing with {} using code {}...", target, code);
            let status = std::process::Command::new("adb")
                .args(["pair", &target, &code])
                .status();

            match status {
                Ok(s) if s.success() => {
                    println!("[+] Successfully paired!");
                    let _ = std::process::Command::new("adb")
                        .args(["tcpip", "5555"])
                        .status();
                    let _ = std::process::Command::new("adb")
                        .args(["kill-server"])
                        .status();
                    println!("[+] radb is now paired and configured permanently on port 5555!");
                }
                _ => {
                    eprintln!("[-] Pairing failed. Ensure pairing dialog is active on screen.");
                }
            }
            return;
        }
        "shell" => {
            let client = match radb::RuriClient::auto_connect() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("[-] Could not connect to local wireless ADB: {}", e);
                    std::process::exit(1);
                }
            };
            println!(
                "[*] Connecting interactive shell to 127.0.0.1:{} (UID 2000)...",
                client.port()
            );
            if let Err(e) = client.open_shell() {
                eprintln!("[-] Shell error: {}", e);
                std::process::exit(1);
            }
            return;
        }
        _ => {}
    }

    let client = match radb::RuriClient::auto_connect() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[-] Could not connect to local wireless ADB: {}", e);
            eprintln!("[-] Ensure Wireless Debugging is enabled in Developer Options.");
            std::process::exit(1);
        }
    };

    match sub_args[0].as_str() {
        "exec" => {
            if sub_args.len() < 2 {
                eprintln!("Usage: stella radb exec <command>");
                std::process::exit(1);
            }
            let cmd = sub_args[1..].join(" ");
            if let Err(e) = client.exec_stream(&cmd) {
                eprintln!("[-] Exec error: {}", e);
                std::process::exit(1);
            }
        }
        "push" => {
            if sub_args.len() < 3 {
                eprintln!("Usage: stella radb push <local_file> <remote_destination>");
                std::process::exit(1);
            }
            let local_path = std::path::Path::new(&sub_args[1]);
            let remote_dest = &sub_args[2];
            println!("[*] Pushing {} -> {}...", local_path.display(), remote_dest);
            if let Err(e) = client.push_file(local_path, remote_dest) {
                eprintln!("[-] Push failed: {}", e);
                std::process::exit(1);
            }
            println!("[+] Successfully pushed to {}", remote_dest);
        }
        "pull" => {
            if sub_args.len() < 3 {
                eprintln!("Usage: stella radb pull <remote_file> <local_destination>");
                std::process::exit(1);
            }
            let remote_file = &sub_args[1];
            let local_path = std::path::Path::new(&sub_args[2]);
            println!("[*] Pulling {} -> {}...", remote_file, local_path.display());
            if let Err(e) = client.pull_file(remote_file, local_path) {
                eprintln!("[-] Pull failed: {}", e);
                std::process::exit(1);
            }
            println!("[+] Successfully pulled to {}", local_path.display());
        }
        "scan" => {
            println!("[+] Active wireless ADB port: {}", client.port());
        }
        _other => {
            let cmd = sub_args.join(" ");
            if let Err(e) = client.exec_stream(&cmd) {
                eprintln!("[-] Exec error: {}", e);
                std::process::exit(1);
            }
        }
    }
}
