import timeit
from tkinter import * 
import matplotlib.pyplot as plt


def bench_mmm(startn,maxn,step,loops):
    count=0
    
    #Preallocate results lists
    avg_gflops = int((1+(maxn-startn)/step))*[0] 
    peak_gflops = int((1+(maxn-startn)/step))*[0]
    raw_times = [int(loops)*[0] for i in range(int(1+(maxn-startn)/step))]
    all_gflops = [int(loops)*[0] for i in  range(int(1+(maxn-startn)/step))]
    mat_size=int((1+(maxn-startn)/step))*[0] 

    for n in range(startn,maxn+step,step):
        setup_string = "from pylab import rand,dot;n=%d;a=rand(n,n);b=rand(n,n)" % n
        time_list = timeit.repeat("a.dot(b)", setup=setup_string, repeat=loops,number=1)
        raw_times[count] = time_list
        total_time = sum(time_list)
        avg_time = total_time / loops
        peak_time = min(time_list)
        num_ops = 2*n**3-n**2
        avg_gflops[count] = (num_ops/avg_time)/10**9
        peak_gflops[count] = (num_ops/peak_time)/10**9
        all_gflops[count] = [(num_ops/time)/10**9 for time in raw_times[count]]        
        mat_size[count] = n
        count=count+1
    
    plt.plot(mat_size,avg_gflops,'*-',label="Average over %d runs" %loops)
    plt.plot(mat_size,peak_gflops,'*-',label="Peak")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Matrix Size');
    plt.ylabel('GFlop/s');
    plt.show()
    
    return(max(peak_gflops),raw_times,all_gflops)

peak_flops = bench_mmm(250,2000,250,5)
#Maximum flops found
print(peak_flops[0])
