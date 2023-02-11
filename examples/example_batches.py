# example_batches.py
# Authors: Rasmus Orsoe, Stephan Meighen-Berger
# Example for python multithreading
# imports
from multiprocessing import Process, Pipe
import time


def simulate_batch(settings):
    id, n_events = settings
    import sys
    sys.path.append('../')
    from prometheus import Prometheus, config
    from jax.config import config as jconfig
    import gc
    jconfig.update("jax_enable_x64", True)
    config["general"]["random state seed"] = id
    config["general"]["meta_name"] = 'meta_data_%d' % id
    config['general']['clean up'] = False
    config['lepton injector']['simulation']['output name'] = "./output/custom_%d_output_LI.h5" % (id)
    config['photon propagator']['storage location'] = './output/custom_%d_' % (id)
    config['lepton injector']['simulation']['nevents'] = n_events
    config['lepton injector']['simulation']['minimal energy'] = 1e1
    config['lepton injector']['simulation']['maximal energy'] = 1e2
    config['lepton injector']['simulation']["injection radius"] = 150
    config['lepton injector']['simulation']["endcap length"] = 200
    config['lepton injector']['simulation']["cylinder radius"] = 150
    config['lepton injector']['simulation']["cylinder height"] = 400
    config['detector']['injection offset'] = [0., 0., 0]
    config['photon propagator']['name'] = 'olympus'
    config["detector"]['new detector'] = True
    config["detector"]['geo file'] = '../prometheus/data/custom.txt'
    config["detector"]["file name"] = '../prometheus/data/custom-f2k'
    prometheus = Prometheus()
    prometheus.sim()
    del prometheus
    gc.collect()
    return

def distribute_jobs(n_workers, n_events):
    # create a list to keep all processes
    processes = []
    # create a list to keep connections
    parent_connections = []
    for i in range(n_workers):
        settings = [i, int(n_events / n_workers)]
        # create a pipe for communication
        parent_conn, _ = Pipe()
        parent_connections.append(parent_conn)
        # create the process, pass instance and connection
        process = Process(
            target=simulate_batch, args=(settings, )
        )
        processes.append(process)
    # start all processes
    for process in processes:
        process.start()
    # make sure that all processes have finished
    for process in processes:
        process.join()
    instances_total = 0
    for parent_connection in parent_connections:
        instances_total += parent_connection.recv()[0]
    return instances_total
    #simulate_batch(settings[0])

if __name__ == "__main__":
    start_time = time.time()
    n_workers = 2
    n_events = 100
    processes = distribute_jobs(n_workers, n_events)
    print("Finished processes: %d" % processes)
    time_diff = (time.time() - start_time) / 60
    print(
        'simulated %s events in %s minutes using %s cores. \n Time pr. event pr. core is: %s' % (
                n_events, time_diff, n_workers, (n_events/n_workers)/time_diff
            )
        )
