from rag_app.database import get_dynamic_dashboard_metrics
import json

def test():
    result = get_dynamic_dashboard_metrics()
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test()
