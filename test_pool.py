import multiprocessing
import time

def complex_cpu_bound_task(x):
    # Perform some complex CPU-bound computation
    result = 0
    for i in range(x):
        for j in range(x):
            for k in range(x):
                for kk in range(x):
                    result += i * j * k * kk
    return result

if __name__ == '__main__':
    # Create a Pool with 32 workers
    with multiprocessing.Pool(processes=32) as pool:
        # Generate a list of input values
        input_data = range(1000, 11000, 1000)

        # Measure the start time
        start_time = time.time()

        # Apply the complex CPU-bound task to the input data using Pool.map
        results = pool.map(complex_cpu_bound_task, input_data)

        # Measure the end time
        end_time = time.time()

    # Print the results and elapsed time
    print(results)
    print("Elapsed time:", end_time - start_time, "seconds")
