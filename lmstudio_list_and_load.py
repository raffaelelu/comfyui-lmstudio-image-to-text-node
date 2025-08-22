import lmstudio as lms, pprint, traceback, sys
# Heuristic exclude keywords for large models
exclude = ['20b','30b','65b','70b','100b','13b','35b','40b','60b','220b']

print('Top-level list_downloaded_models exists:', hasattr(lms, 'list_downloaded_models'))
try:
    dd = lms.list_downloaded_models()
    print('lms.list_downloaded_models():')
    pprint.pprint(dd)
except Exception as e:
    print('lms.list_downloaded_models error:', e)

try:
    with lms.Client() as client:
        print('\nclient.list_downloaded_models:')
        try:
            downloaded = client.list_downloaded_models()
            pprint.pprint(downloaded)
        except Exception as e:
            print('client.list_downloaded_models error:', e)
            downloaded = []

        try:
            llm_only = client.llm.list_downloaded()
            print('\nclient.llm.list_downloaded():')
            pprint.pprint(llm_only)
        except Exception as e:
            print('client.llm.list_downloaded error:', e)
            llm_only = []

        # Choose a candidate
        candidates = downloaded or llm_only
        print('\nFound', len(candidates), 'downloaded models')
        chosen = None
        for mdl in candidates:
            # model object may be a DownloadedLlm with model_key/display_name
            try:
                key = getattr(mdl, 'model_key', None) or getattr(mdl, 'id', None) or str(mdl)
            except Exception:
                key = str(mdl)
            name = getattr(mdl, 'display_name', None) or key
            print('Candidate:', key, '->', name)
            low = key.lower() if key else ''
            if not any(k in low for k in exclude):
                chosen = (mdl, key, name)
                break

        if not chosen:
            print('No small candidate found by heuristic, will not attempt to load automatically.')
            sys.exit(0)

        mdl_obj, key, name = chosen
        print('\nAttempting to load candidate:', key, name)

        # Try using downloaded.model() if available
        try:
            if hasattr(mdl_obj, 'model') and callable(mdl_obj.model):
                print('Using downloaded.model() method on the Downloaded model object')
                model_handle = mdl_obj.model()
                print('downloaded.model() returned:', type(model_handle), model_handle)
            else:
                print('downloaded.model() not available; trying client.llm.model(key)')
                model_handle = client.llm.model(key)
                print('client.llm.model returned:', type(model_handle), model_handle)

            print('\nNow checking client.llm.list_loaded():')
            try:
                print(client.llm.list_loaded())
            except Exception as e:
                print('client.llm.list_loaded() error:', e)

        except Exception as e:
            print('Error while loading model:', e)
            traceback.print_exc()
except Exception as e:
    print('Client() error:')
    traceback.print_exc()
