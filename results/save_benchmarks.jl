using JSON, CSV, DataFrames

function jsons_to_csv(json_files, output_csv)
    # Placeholder DataFrame initialization
    df = DataFrame()
    first_file = true
    kernel_names = []

    for file in json_files
        if isfile(file)
            # Read the JSON file
            contents = JSON.parsefile(file)

            kernel_name = splitext(basename(file))[1]
            push!(kernel_names, kernel_name)

            # For the first file, initialize the DataFrame with column names and types
            if first_file
                for (key, value) in contents
                    df[!, Symbol(key)] = Vector{typeof(value)}()
                end
                first_file = false
            end
            
            # Create a new row for the DataFrame
            row = [contents[key] for key in keys(contents)]
            
            # Append the row to the DataFrame
            push!(df, row)
        else
            println("File does not exist: ", file)
        end
    end

    kernel_names = replace.(kernel_names, "_" => "-")
    df = insertcols!(df, 1, :KernelName => kernel_names)

    if !isdir("csv")
        mkdir("csv")
    end

    # Write the DataFrame to a CSV file
    CSV.write("csv/" * output_csv, df)
end

function generate_benchmark_csv(benchmark, suffix)
    jsons_to_csv([suffix * "/" * benchmark * ".json"], benchmark * "_" * suffix * ".csv")
end


function generate_benchmarks_csv(suffix)
    generate_benchmark_csv("dicf", suffix)
    generate_benchmark_csv("dicfs", suffix)
    generate_benchmark_csv("dcauchy", suffix)
    generate_benchmark_csv("dtrpcg", suffix)
    generate_benchmark_csv("dprsrch", suffix)
    generate_benchmark_csv("daxpy", suffix)
    generate_benchmark_csv("dssyax", suffix)
    generate_benchmark_csv("dmid", suffix)
    generate_benchmark_csv("dgpstep", suffix)
    generate_benchmark_csv("dbreakpt", suffix)
    generate_benchmark_csv("dnrm2", suffix)
    generate_benchmark_csv("nrm2", suffix)
    generate_benchmark_csv("dcopy", suffix)
    generate_benchmark_csv("ddot", suffix)
    generate_benchmark_csv("dscal", suffix)
    generate_benchmark_csv("dtrqsol", suffix)
    generate_benchmark_csv("dspcg", suffix)
    generate_benchmark_csv("dgpnorm", suffix)
    generate_benchmark_csv("dtron", suffix)
    generate_benchmark_csv("driver_kernel", suffix)
end

generate_benchmarks_csv("cuda")
generate_benchmarks_csv("gen")