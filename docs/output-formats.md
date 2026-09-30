# Output Formats

VoxPipe can render human-readable transcripts and a machine-readable JSON transcript.

Supported human-readable formats:

- Markdown
- plain text
- HTML
- SRT

## JSON schema

Example:

```json
{
  "language": "russian",
  "language_probability": 1.0,
  "duration": 156.4,
  "segments": [
    {
      "start": 0.4,
      "end": 7.0,
      "text": "Да. Добрый день. Меня зовут Виктория...",
      "words": [
        { "start": 0.42, "end": 0.9, "word": "Да" }
      ]
    }
  ],
  "utterances": [
    {
      "speaker": "SPEAKER_01",
      "start": 0.4,
      "end": 7.0,
      "text": "Да. Добрый день. Меня зовут Виктория..."
    },
    {
      "speaker": "SPEAKER_00",
      "start": 7.0,
      "end": 10.1,
      "text": "Ну да, слушаю."
    }
  ],
  "text": "SPEAKER_01: Да. Добрый день...\nSPEAKER_00: Ну да, слушаю."
}
```

`language` uses whisper.cpp's language name, for example `russian`, rather than an ISO shorthand such as `ru`.

The schema was kept compatible with the earlier Transcriber application.

## Word timestamps

In watch mode, JSON always contains word timestamps because it is the durable archive.

For one-shot CLI JSON, word timestamps are included only when `--words` is supplied.

On the original test recording, this changed JSON size from approximately 18 KB to 96 KB.

whisper.cpp word `probability` values are not copied into the VoxPipe JSON because the old application did not store them either.

## Speaker labels

Speaker labels are assigned by clustering order:

```text
SPEAKER_00
SPEAKER_01
...
```

They are stable only inside a single file.

VoxPipe does not perform persistent speaker identity recognition.

## Markdown

Default paragraph layout:

```markdown
**SPEAKER_00** · [00:00 – 00:03]

Good morning everyone. Let's start the review.
```

Sentence layout:

```markdown
- [00:00 – 00:03] **SPEAKER_00**: Good morning everyone.
- [00:00 – 00:03] **SPEAKER_00**: Let's start the review.
```

Long turns are split at sentence boundaries to avoid a wall of text.

## Plain text

Plain text can optionally include timestamps with:

```bash
voxpipe meeting.mp3 --format txt --timestamps
```

## HTML

HTML output is self-contained and color-codes speakers with separate accent colors.

It can be opened directly in a browser.

## SRT

SRT output splits long turns into short subtitle captions.

The original target is approximately 5-7 words per caption.

Control the target with:

```bash
--caption-words N
```

The turn time span is divided proportionally across captions so subtitle timing remains smooth.

## CLI examples

```bash
voxpipe meeting.mp3
voxpipe meeting.mp3 -o meeting.md
voxpipe meeting.mp3 -o meeting.txt --format txt --layout sentence
voxpipe meeting.mp3 -o meeting.html --format html
voxpipe meeting.mp3 -o meeting.srt --format srt
voxpipe meeting.mp3 -o meeting.md --json meeting.json --words
voxpipe meeting.mp3 --language en --timestamps
```

A recognized extension in `-o FILE` selects the corresponding output format.
