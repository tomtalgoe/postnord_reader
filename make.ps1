param([string]$command, [string]$arg, [string]$arg2)

function Stop-Server {
    Write-Host "Stopping existing server..."
    
    # Find the process running `api.py` and terminate it
    $process = Get-Process | Where-Object { $_.ProcessName -match "python" -and $_.Path -match "api.py" }

    if ($process) {
        $process | Stop-Process -Force
        Write-Host "Server stopped."
    } else {
        Write-Host "No running server found."
    }
}

switch ($command) {
    "start" {
        Write-Host "Starting server..."
        virtualenv/Scripts/activate
        Start-Process -NoNewWindow -FilePath "python.exe" -ArgumentList "api.py" -RedirectStandardOutput "../server.log" -RedirectStandardError "../server_err.log"
        Write-Host "Server started."
    }
    "restart" {
        Stop-Server
        Write-Host "Restarting server..."
        virtualenv/Scripts/activate
        Start-Process -NoNewWindow -FilePath "python.exe" -ArgumentList "api.py" -RedirectStandardOutput "../server.log" -RedirectStandardError "../server_err.log"
        Write-Host "Server restarted."
    }
    "stop" {
        Stop-Server
    }
    default {
        Write-Host "Invalid command`n`nSupported commands: start, stop, restart"
    }
}