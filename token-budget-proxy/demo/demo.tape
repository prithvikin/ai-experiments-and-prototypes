# Regenerate the README GIF:  vhs demo/demo.tape
#
# Runs the demo agent against the mock upstream and shows the waste report.
# Trimmed to the two costliest findings so the whole thing fits on one screen --
# the headline count still reports all six.

Output demo/demo.gif

Set Shell zsh
Set FontSize 13
Set Width 1240
Set Height 665
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms

# Setup happens off-camera: start the proxy, make sure the ledger is empty.
Hide
Type "cd /Users/prithvikinariwala/Documents/projects/ai-experiments-and-protoypes/token-budget-proxy"
Enter
Type "export PATH=$PWD/.venv/bin:$PATH FORCE_COLOR=1 COLUMNS=118"
Enter
Type "python -m tbp serve --upstream mock --policy policy.observe.yaml >/dev/null 2>&1 &"
Enter
Sleep 3s
Type "python -m tbp reset >/dev/null; clear"
Enter
Show

Sleep 1s
Type "python demo/agent.py sloppy"
Enter
Sleep 3s

Type "python -m tbp report --top 2"
Enter
Sleep 7s

Hide
Type "kill %1 2>/dev/null"
Enter
