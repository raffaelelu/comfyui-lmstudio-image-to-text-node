import lmstudio as lms, traceback, sys
MODEL = "MODELNAME"  # replace with the exact model name you used in the node if you want
print("Top-level has list_loaded_models:", hasattr(lms, "list_loaded_models"))
try:
    print("lms.list_loaded_models():", lms.list_loaded_models())
except Exception as e:
    print("lms.list_loaded_models error:", e)
try:
    with lms.Client() as client:
        print("client.list_loaded_models:", client.list_loaded_models() if hasattr(client, "list_loaded_models") else None)
        print("client.llm.list_loaded() exists:", hasattr(client.llm, "list_loaded"))
        try:
            print("client.llm.list_loaded():", client.llm.list_loaded())
        except Exception as e:
            print("client.llm.list_loaded() error:", e)
        try:
            print('Trying client.llm.model(MODEL)...')
            m = client.llm.model(MODEL)
            print('Model loaded OK:', type(m), getattr(m, "__class__", None))
        except Exception as e:
            print('client.llm.model error:')
            traceback.print_exc()
except Exception as e:
    print("Client() error:")
    traceback.print_exc()
