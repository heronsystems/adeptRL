"""
parse env, actor, network
create N rollout_workers
create env, network, learner
broadcast weights and wait

if sync:
    while not done:
        get rollout caches from all workers
        combine (possible to device)
        train
        send weights & wait
elif async:
    while not done:
        t1:
            get rollout caches into queue 
        t2:
            when len(cache) >= train size:
                combine?
                train
                send weights
"""
