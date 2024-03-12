using BenchmarkTools
using JSON
using Statistics

function aggregate_benchmarks(benchmarks)
    combined_times = []
    combined_gctimes = []
    combined_memory = 0
    combined_allocs = 0
    for b in benchmarks
        append!(combined_times, b.times)
        append!(combined_gctimes, b.gctimes)
        combined_allocs += b.allocs
        combined_memory += b.memory
    end
    return BenchmarkTools.Trial(BenchmarkTools.Parameters(; samples=length(combined_times), evals=1), combined_times, combined_gctimes, combined_memory, combined_allocs)
end

function save_benchmark(benchmark, filename)
    b_dict = Dict("max" => maximum(benchmark).time / 1000,
     "min" => minimum(benchmark).time / 1000,
     "mean" => Statistics.mean(benchmark).time / 1000,
     "var" => Statistics.var(benchmark).time / 1000,
     "std" => Statistics.std(benchmark).time / 1000,
     "samples" => length(benchmark.times),
     "memory" => benchmark.memory,
     "allocs" => benchmark.allocs)

    json_string = JSON.json(b_dict)

    open(filename,"w") do f 
        write(f, json_string) 
    end
end
