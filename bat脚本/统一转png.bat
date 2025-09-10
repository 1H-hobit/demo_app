@echo off
setlocal enabledelayedexpansion

for %%f in (*) do (
    if not "%%f"=="%~nx0" (
        set "filename=%%~nf"
        ren "%%f" "!filename!.png"
    )
)