import asyncio
import json
import sys
import traceback

async def test():
    try:
        import main
        print("Importing main... OK")

        await main._startup()
        print(f"KB loaded: {len(main._kb[0])} entries")
        print(f"Model loaded: {main._model is not None}")

        from main import WebRequest
        req = WebRequest(
            text=(
                "You agree that we may share your data with third parties without notice. "
                "We may terminate your account at any time for any reason. "
                "We are not liable for any damages arising from use of this service."
            ),
            threshold=0.65,
        )
        result = await main._analyze(req)
        print(f"risk_score : {result['risk_score']}")
        print(f"flags count: {len(result['flags'])}")
        print(f"summary    : {result['summary'][:300]}")
        if result["flags"]:
            f = result["flags"][0]
            print(f"top flag   : [{f['severity']}] {f['category']} score={f['score']}")
        print("\nALL CHECKS PASSED")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

asyncio.run(test())
