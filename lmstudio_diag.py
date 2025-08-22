import lmstudio as lms
print("Top-level attributes:", [a for a in dir(lms) if not a.startswith("_")])
try:
    print("lms.list_loaded_models():", getattr(lms, "list_loaded_models", None) and lms.list_loaded_models())
except Exception as e:
    print("list_loaded_models error:", e)
try:
    with lms.Client() as client:
        print("Client attrs:", [a for a in dir(client) if not a.startswith("_")])
        try:
            print("Client.llm attrs:", [a for a in dir(client.llm) if not a.startswith("_")])
        except Exception as e:
            print("client.llm attr error:", e)
        try:
            val = client.list_loaded_models() if hasattr(client, 'list_loaded_models') else None
            print("client.list_loaded_models():", val)
        except Exception as e:
            print("client.list_loaded_models error:", e)
except Exception as e:
    print("Client() error:", e)
