import lmstudio as lms, traceback, sys, time
MODEL = "gemma-3-4b-it-qat-abliterated-i1"
print('Attempting to load model:', MODEL)
try:
    with lms.Client() as client:
        try:
            m = client.llm.model(MODEL)
            print('Loaded model handle type:', type(m))
        except Exception as e:
            print('client.llm.model error:')
            traceback.print_exc()

        try:
            print('\nclient.llm.list_loaded():')
            print(client.llm.list_loaded())
        except Exception as e:
            print('client.llm.list_loaded() error:')
            traceback.print_exc()

        # keep client open a short moment so server logs appear cleanly
        time.sleep(0.5)
except Exception as e:
    print('Client() error:')
    traceback.print_exc()
