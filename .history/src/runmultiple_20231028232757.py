import multiprocessing
import numpy as np
import nn

def run_main(arg):
    nn.main(arg)

if __name__ == '__main__':

    arguments = np.array([
        "validation_one('Estar', 'TI33_250a', 'TI33_500a', 'Exp', 1, lay = 9, wid = 32)",'''
        "validation_one('Estar', 'TI33_250a', 'TI33_500a', 'Exp', 2, lay = 9, wid = 32)",
        "validation_one('Estar', 'TI33_250a', 'TI33_500a', 'Exp', 3, lay = 9, wid = 32)",
        "validation_one('Estar', 'TI33_250a', 'TI33_500a', 'Exp', 4, lay = 9, wid = 32)",
        "validation_one('Estar', 'TI33_250a', 'TI33_500a', 'Exp', 5, lay = 9, wid = 32)",
        "validation_one('Estar', 'TI33_250a', 'TI33_500a', 'Exp', 10, lay = 9, wid = 32)",
        "validation_one('Estar', 'TI33_250a', 'TI33_500a', 'Exp', 20, lay = 9, wid = 32)",
        "validation_one('sigma_y', 'TI33_250a', 'TI33_500a', 'Exp', 1, lay = 9, wid = 32)",
        "validation_one('sigma_y', 'TI33_250a', 'TI33_500a', 'Exp', 2, lay = 9, wid = 32)",
        "validation_one('sigma_y', 'TI33_250a', 'TI33_500a', 'Exp', 3, lay = 9, wid = 32)",
        "validation_one('sigma_y', 'TI33_250a', 'TI33_500a', 'Exp', 4, lay = 9, wid = 32)",
        "validation_one('sigma_y', 'TI33_250a', 'TI33_500a', 'Exp', 5, lay = 9, wid = 32)",
        "validation_one('sigma_y', 'TI33_250a', 'TI33_500a', 'Exp', 10, lay = 9, wid = 32)",'''
        "validation_one('sigma_y', 'TI33_250a', 'TI33_500a', 'Exp', 20, lay = 9, wid = 32)"
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