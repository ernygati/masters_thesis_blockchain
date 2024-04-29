def get(config):
    #set blockchain params to change
    params = ['NUM_THREADS',
              'DEFAULT_TICKS_PER_SLOT',
              'ITER_BATCH_SIZE',
              'RECV_BATCH_MAX_CPU',
              'DEFAULT_HASHES_PER_SECOND',
              'DEFAULT_TICKS_PER_SECOND']

    out_in_keys = {}
    #gettin outter keys to get access to inner dicts
    for outter in config.keys():
        inner = list(config[outter].keys())
        #print(config[outter].items())
        for p in params:
            if p in inner:
                out_in_keys[p] = outter
    return(out_in_keys)
