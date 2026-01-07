# Troubleshooting Ollama Connection

If you are running the `mlcs` app in WSL (Linux) and Ollama on Windows, you might encounter connection errors like `Connection refused` or `ConnectTimeoutError`.

## 1. Verify Ollama is Running
On Windows, check if the Ollama icon is in your system tray.
Open PowerShell and run:
```powershell
netstat -an | findstr 11434
```
You **must** see a line like `TCP 0.0.0.0:11434` or `TCP 127.0.0.1:11434`. If you see nothing, Ollama is not running.

## 2. Expose to Network
By default, Ollama only listens on `localhost`. To allow WSL to see it:
1.  Click the Ollama tray icon -> **Settings**.
2.  Enable **"Expose Ollama to the network"**.
3.  **Restart Ollama** (Quit and start again).

## 3. Configure the App
In your WSL terminal, you need to tell the app where Ollama is.
1.  Find your Windows IP:
    ```bash
    ip route | grep default
    # The IP listed as 'via X.X.X.X' is usually your Windows host (e.g., 172.x.x.1)
    ```
2.  Set the environment variable:
    ```bash
    export OLLAMA_HOST=http://YOUR_WINDOWS_IP:11434
    ```

## 4. Firewall
If it still fails, Windows Firewall might be blocking the connection.
Run this in **PowerShell (Admin)**:
```powershell
New-NetFirewallRule -DisplayName "Ollama" -Direction Inbound -LocalPort 11434 -Protocol TCP -Action Allow
```
