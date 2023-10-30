import multiprocessing
import numpy as np
import nn

def run_main(arg):
    nn.main(arg)

if __name__ == '__main__':

    arguments = np.array([
        "validation_one('sigma_y', 'TI33_250a', 'TI33_250a', 'Exp', 1)",
        "validation_one('sigma_y', 'Ti33_250a', 'TI33_250a', 'Exp', 2)",
        "validation_one('sigma_y', 'Ti33_250a', 'TI33_250a', 'Exp', 3)",
        "validation_one('sigma_y', 'Ti33_250a', 'TI33_250a', 'Exp', 4)",
        "validation_one('sigma_y', 'Ti33_250a', 'TI33_250a', 'Exp', 5)",
        "validation_one('sigma_y', 'Ti33_250a', 'TI33_250a', 'Exp', 10)",
        "validation_one('sigma_y', 'Ti33_250a', 'TI33_250a', 'Exp', 20)",
        "validation_one('Estar', 'Ti33_250a', 'TI33_250a', 'Exp', 1)",
        "validation_one('Estar', 'Ti33_250a', 'TI33_250a', 'Exp', 2)",
        "validation_one('Estar', 'Ti33_250a', 'TI33_250a', 'Exp', 3)",
        "validation_one('Estar', 'Ti33_250a', 'TI33_250a', 'Exp', 4)",
        "validation_one('Estar', 'Ti33_250a', 'TI33_250a', 'Exp', 5)",
        "validation_one('Estar', 'Ti33_250a', 'TI33_250a', 'Exp', 10)",
        "validation_one('Estar', 'Ti33_250a', 'TI33_250a', 'Exp', 20)"
        ])
    
    processes = []
    num_processes = len(arguments)
    for i in range(num_processes):        
        process = multiprocessing.Process(target=run_main, args=(arguments[i],))
        processes.append(process)

    for process in processes:
        process.start()
    for process in processes:
        process.join()