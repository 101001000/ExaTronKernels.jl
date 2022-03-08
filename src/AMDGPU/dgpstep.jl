@inline function dgpstep(n::Int,x::CuDeviceArray{Float64,1},xl::CuDeviceArray{Float64,1},
                         xu::CuDeviceArray{Float64,1},alpha,w::CuDeviceArray{Float64,1},
                         s::CuDeviceArray{Float64,1})
    tx = threadIdx().x
    ty = threadIdx().y

    if tx <= n && ty == 1
        @inbounds begin
            # It might be better to process this using just a single thread,
            # rather than diverging between multiple threads.

            if x[tx] + alpha*w[tx] < xl[tx]
                s[tx] = xl[tx] - x[tx]
            elseif x[tx] + alpha*w[tx] > xu[tx]
                s[tx] = xu[tx] - x[tx]
            else
                s[tx] = alpha*w[tx]
            end
        end
    end
    AMDGPU.sync_workgroup()

    return
end
