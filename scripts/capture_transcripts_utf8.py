from pathlib import Path
from config import validate
from stt.sarvam_client import transcribe

SAMPLES = [Path('data/test_h1.wav'), Path('data/test_h2.wav'), Path('data/test_h3.wav')]

def main():
    try:
        validate()
    except Exception as e:
        print('Config validation failed:', e)
        return

    out_path = Path('scripts/transcripts_utf8.txt')
    with out_path.open('w', encoding='utf-8') as out:
        for p in SAMPLES:
            out.write(f'FILE: {p}\n')
            res = transcribe(str(p))
            if res.get('success'):
                out.write(f'LATENCY_MS: {res.get("latency_ms"):.1f}\n')
                out.write('TRANSCRIPT:\n')
                out.write(res.get('transcript') + '\n')
            else:
                out.write(f'FAILED: {res.get("error")}\n')
            out.write('\n')
    print('Wrote', out_path)

if __name__ == "__main__":
    main()
