tomomibot
---

```
▄▄▄▄▄      • ▌ ▄ ·.       • ▌ ▄ ·. ▪  ▄▄▄▄·      ▄▄▄▄▄
•██  ▪     ·██ ▐███▪▪     ·██ ▐███▪██ ▐█ ▀█▪▪    •██
 ▐█.▪ ▄█▀▄ ▐█ ▌▐▌▐█· ▄█▀▄ ▐█ ▌▐▌▐█·▐█·▐█▀▀█▄ ▄█▀▄ ▐█.▪
 ▐█▌·▐█▌.▐▌██ ██▌▐█▌▐█▌.▐▌██ ██▌▐█▌▐█▌██▄▪▐█▐█▌.▐▌▐█▌·
 ▀▀▀  ▀█▄▀▪▀▀  █▪▀▀▀ ▀█▄▀▪▀▀  █▪▀▀▀▀▀▀·▀▀▀▀  ▀█▄▀▪▀▀▀
```

Artificial intelligence bot for live voice improvisation.

### Installation

```
git clone git@github.com:adzialocha/tomomibot.git
cd tomomibot
pip install -e .
```

### Usage

```
Usage: tomomibot [OPTIONS] COMMAND [ARGS]...

  Artificial intelligence bot for live voice
  improvisation.

Options:
  -v, --verbose  Enables verbose mode.
  --help         Show this message and exit.

Commands:
  generate  Generate voice based on .wav file
  start     Start a live session
  status    Display system info and audio devices
  train     Train a model for sequence prediction
```

### Known issues

* `error during recording: -10863` (MacOS): Selected samplerate differs from the one of your recording device, change it via the `--samplerate` argument of the `start` command.

### License

`MIT`
