Remove-Item Env:HTTP_PROXY -ErrorAction Ignore
Remove-Item Env:HTTPS_PROXY -ErrorAction Ignore

[Environment]::SetEnvironmentVariable("HTTP_PROXY", $null, "User")
[Environment]::SetEnvironmentVariable("HTTPS_PROXY", $null, "User")

Write-Output "Proxy desactive"