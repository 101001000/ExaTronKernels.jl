using CUDA
using ExaTronKernels
using LinearAlgebra
using Random
using Test
using BenchmarkTools

include("../src/utils.jl")

try
    tmp = CuArray{Float64}(undef, 10)
catch e
    throw(e)
end


# Test ExaTron's internal routines written for GPU.
# 
# The current implementation assumes the following:
#   - A thread block takes a matrix structure, (tx,ty), with
#     n <= blockDim().x = blockDim().y <= 32 and blockDim().x is even.
#   - Arguments passed on to these routines are assumed to be
#     of size at least n. This is to prevent multiple thread
#     divergence when we call a function with n_hat < n.
#     Such a case occurs when we fix active variables.
# 
# We test the following routines, where [O] indicates if the routine
# is checked if n < blockDim().x is OK.
#   - dicf     [O][O]: this routine also tests dnsol and dtsol.
#   - dicfs    [O][T]
#   - dcauchy  [O][T]
#   - dtrpcg   [O][T]
#   - dprsrch  [O][T]
#   - daxpy    [O][O]
#   - dssyax   [O][O]: we do shuffle using blockDim().x.
#   - dmid     [O][O]: we could use a single thread only to multiple divergences.
#   - dgpstep  [O][O]
#   - dbreakpt [O][O]: we use the existing ExaTron implementation as is.
#   - dnrm2    [O][O]: we do shuffle using blockDim().x.
#   - nrm2     [O][O]: we do shuffle using blockDim().x.
#   - dcopy    [O][O]
#   - ddot     [O][O]
#   - dscal    [O][O]
#   - dtrqsol  [O][O]
#   - dspcg    [O][T]: we use a single thread to avoid multiple divergences.
#   - dgpnorm  [O][O]
#   - dtron    [O]
#   - driver_kernel


Random.seed!(0)
itermax = 1
const n = 4
nblk = 4

@inline function daxpy(n::Int, da::Float64,
    dx::CuDeviceArray{Float64,1}, incx::Int,
    dy::CuDeviceArray{Float64,1}, incy::Int)
    tx = threadIdx().x

    if tx <= n
        @inbounds dy[tx] = dy[tx] + da * dx[tx]
    end
    CUDA.sync_threads()

    return
end

@inline function dbreakpt(n::Int, x::CuDeviceArray{Float64,1}, xl::CuDeviceArray{Float64,1},
    xu::CuDeviceArray{Float64,1}, w::CuDeviceArray{Float64,1})
    zero = 0.0
    nbrpt = 0
    brptmin = zero
    brptmax = zero

    @inbounds for i = 1:n
        if (x[i] < xu[i] && w[i] > zero)
            nbrpt = nbrpt + 1
            brpt = (xu[i] - x[i]) / w[i]
            if nbrpt == 1
                brptmin = brpt
                brptmax = brpt
            else
                brptmin = min(brpt, brptmin)
                brptmax = max(brpt, brptmax)
            end
        elseif (x[i] > xl[i] && w[i] < zero)
            nbrpt = nbrpt + 1
            brpt = (xl[i] - x[i]) / w[i]
            if nbrpt == 1
                brptmin = brpt
                brptmax = brpt
            else
                brptmin = min(brpt, brptmin)
                brptmax = max(brpt, brptmax)
            end
        end
    end

    # Handle the exceptional case.

    if nbrpt == 0
        brptmin = zero
        brptmax = zero
    end
    CUDA.sync_threads()

    return nbrpt, brptmin, brptmax
end

@inline function dcauchy(n::Int, x::CuDeviceArray{Float64,1},
    xl::CuDeviceArray{Float64}, xu::CuDeviceArray{Float64,1},
    A::CuDeviceArray{Float64,2}, g::CuDeviceArray{Float64,1},
    delta::Float64, alpha::Float64, s::CuDeviceArray{Float64,1},
    wa::CuDeviceArray{Float64,1})
    p5 = 0.5
    one = 1.0

    # Constant that defines sufficient decrease.

    mu0 = 0.01

    # Interpolation and extrapolation factors.

    interpf = 0.1
    extrapf = 10.0

    # Find the minimal and maximal break-point on x - alpha*g.

    dcopy(n, g, 1, wa, 1)
    dscal(n, -one, wa, 1)
    nbrpt, brptmin, brptmax = dbreakpt(n, x, xl, xu, wa)

    # Evaluate the initial alpha and decide if the algorithm
    # must interpolate or extrapolate.

    dgpstep(n, x, xl, xu, -alpha, g, s) # s = P(x - alpha*g) - x
    if dnrm2(n, s, 1) > delta
        interp = true
    else
        dssyax(n, A, s, wa)
        gts = ddot(n, g, 1, s, 1)
        q = p5 * ddot(n, s, 1, wa, 1) + gts
        interp = (q >= mu0 * gts)
    end

    # Either interpolate or extrapolate to find a successful step.

    if interp

        # Reduce alpha until a successful step is found.

        search = true
        while search

            # This is a crude interpolation procedure that
            # will be replaced in future versions of the code.

            alpha = interpf * alpha
            dgpstep(n, x, xl, xu, -alpha, g, s)
            if dnrm2(n, s, 1) <= delta
                dssyax(n, A, s, wa)
                gts = ddot(n, g, 1, s, 1)
                q = p5 * ddot(n, s, 1, wa, 1) + gts
                search = (q > mu0 * gts)
            end
        end

    else

        # Increase alpha until a successful step is found.

        search = true
        alphas = alpha
        while (search && alpha <= brptmax)

            # This is a crude extrapolation procedure that
            # will be replaced in future versions of the code.

            alpha = extrapf * alpha
            dgpstep(n, x, xl, xu, -alpha, g, s)
            if dnrm2(n, s, 1) <= delta
                dssyax(n, A, s, wa)
                gts = ddot(n, g, 1, s, 1)
                q = p5 * ddot(n, s, 1, wa, 1) + gts
                if q < mu0 * gts
                    search = true
                    alphas = alpha
                end
            else
                search = false
            end
        end

        # Recover the last successful step.

        alpha = alphas
        dgpstep(n, x, xl, xu, -alpha, g, s)
    end

    return alpha
end

@inline function dcopy(n::Int, dx::CuDeviceArray{Float64,1}, incx::Int,
    dy::CuDeviceArray{Float64,1}, incy::Int)
    tx = threadIdx().x

    # Ignore incx and incy for now.
    if tx <= n
        @inbounds dy[tx] = dx[tx]
    end
    CUDA.sync_threads()

    return
end

@inline function ddot(n::Int, dx::CuDeviceArray{Float64,1}, incx::Int,
    dy::CuDeviceArray{Float64,1}, incy::Int)
    # Currently, all threads compute the same dot product,
    # hence, no sync_threads() is needed.
    # For very small n, we may want to gauge how much gains
    # we could get by run it in parallel.

    v = 0.0
    @inbounds for i = 1:n
        v += dx[i] * dy[i]
    end
    CUDA.sync_threads()
    return v
end

@inline function dgpstep(n::Int, x::CuDeviceArray{Float64,1}, xl::CuDeviceArray{Float64,1},
    xu::CuDeviceArray{Float64,1}, alpha, w::CuDeviceArray{Float64,1},
    s::CuDeviceArray{Float64,1})
    tx = threadIdx().x

    if tx <= n
        @inbounds begin
            # It might be better to process this using just a single thread,
            # rather than diverging between multiple threads.

            if x[tx] + alpha * w[tx] < xl[tx]
                s[tx] = xl[tx] - x[tx]
            elseif x[tx] + alpha * w[tx] > xu[tx]
                s[tx] = xu[tx] - x[tx]
            else
                s[tx] = alpha * w[tx]
            end
        end
    end
    CUDA.sync_threads()

    return
end

@inline function dgpnorm(n::Int, x::CuDeviceArray{Float64,1}, xl::CuDeviceArray{Float64,1},
    xu::CuDeviceArray{Float64,1}, g::CuDeviceArray{Float64,1})
    tx = threadIdx().x

    v = 0.0
    if tx <= n
        @inbounds begin
            if xl[tx] != xu[tx]
                if x[tx] == xl[tx]
                    v = min(g[tx], 0.0)
                elseif x[tx] == xu[tx]
                    v = max(g[tx], 0.0)
                else
                    v = g[tx]
                end

                v = abs(v)
            end
        end
    end

    # shfl_down_sync() will automatically sync threads in a warp.

    offset = 16
    while offset > 0
        v = max(v, CUDA.shfl_down_sync(0xffffffff, v, offset))
        offset >>= 1
    end
    v = CUDA.shfl_sync(0xffffffff, v, 1)

    return v
end

@inline function dgpstep(n::Int, x::CuDeviceArray{Float64,1}, xl::CuDeviceArray{Float64,1},
    xu::CuDeviceArray{Float64,1}, alpha, w::CuDeviceArray{Float64,1},
    s::CuDeviceArray{Float64,1})
    tx = threadIdx().x

    if tx <= n
        @inbounds begin
            # It might be better to process this using just a single thread,
            # rather than diverging between multiple threads.

            if x[tx] + alpha * w[tx] < xl[tx]
                s[tx] = xl[tx] - x[tx]
            elseif x[tx] + alpha * w[tx] > xu[tx]
                s[tx] = xu[tx] - x[tx]
            else
                s[tx] = alpha * w[tx]
            end
        end
    end
    CUDA.sync_threads()

    return
end

@inline function dtsol(n::Int, L::CuDeviceArray{Float64,2},
    r::CuDeviceArray{Float64,1})
    # Solve L'*x = r and store the result in r.

    tx = threadIdx().x

    @inbounds for j = n:-1:1
        if tx == 1
            r[j] = r[j] / L[j, j]
        end
        CUDA.sync_threads()

        if tx < j
            r[tx] = r[tx] - L[tx, j] * r[j]
        end
        CUDA.sync_threads()
    end

    return
end

# Left-looking Cholesky
@inline function dicf(n::Int, L::CuDeviceArray{Float64,2})
    tx = threadIdx().x
    @inbounds for j = 1:n
        # Apply the pending updates.
        if j > 1
            if tx >= j && tx <= n
                for k = 1:j-1
                    L[tx, j] -= L[tx, k] * L[j, k]
                end
            end
        end
        CUDA.sync_threads()

        if (L[j, j] <= 0)
            CUDA.sync_threads()
            return -1
        end

        Ljj = sqrt(L[j, j])
        if tx >= j && tx <= n
            L[tx, j] /= Ljj
        end
        CUDA.sync_threads()
    end

    if tx <= n
        @inbounds for j = 1:n
            if tx > j
                L[j, tx] = L[tx, j]
            end
        end
    end
    CUDA.sync_threads()

    return 0
end

@inline function dicfs(n::Int, alpha::Float64, A::CuDeviceArray{Float64,2},
    L::CuDeviceArray{Float64,2},
    wa1::CuDeviceArray{Float64,1},
    wa2::CuDeviceArray{Float64,1})
    tx = threadIdx().x

    nbmax = 3
    alpham = 1.0e-3
    nbfactor = 512

    zero = 0.0
    one = 1.0
    two = 2.0

    # Compute the l2 norms of the columns of A.
    nrm2!(wa1, A, n)

    # Compute the scaling matrix D.
    if tx <= n
        @inbounds wa2[tx] = (wa1[tx] > zero) ? one / sqrt(wa1[tx]) : one
    end
    CUDA.sync_threads()

    # Determine a lower bound for the step.

    if alpha <= zero
        alphas = alpham
    else
        alphas = alpha
    end

    # Compute the initial shift.

    alpha = zero
    if tx <= n  # No check on ty so that each warp has alpha.
        @inbounds alpha = (A[tx, tx] == zero) ? alphas : max(alpha, -A[tx, tx] * (wa2[tx]^2))
    end

    # shfl_down_sync will automatically sync threads in a warp.

    # Find the maximum alpha in a warp and put it in the first thread.
    #offset = div(blockDim().x, 2)
    offset = 16
    while offset > 0
        alpha = max(alpha, CUDA.shfl_down_sync(0xffffffff, alpha, offset))
        offset >>= 1
    end
    # Broadcast it to the entire threads in a warp.
    alpha = CUDA.shfl_sync(0xffffffff, alpha, 1)

    if alpha > 0
        alpha = max(alpha, alphas)
    end

    # Search for an acceptable shift. During the search we decrease
    # the lower bound alphas until we determine a lower bound that
    # is not acceptaable. We then increase the shift.
    # The lower bound is decreased by nbfactor at most nbmax times.

    nb = 1
    info = 0

    while true
        if tx <= n
            @inbounds for j = 1:n
                L[j, tx] = A[j, tx] * wa2[j] * wa2[tx]
            end
            if alpha != zero
                @inbounds L[tx, tx] += alpha
            end
        end
        CUDA.sync_threads()

        # Attempt a Cholesky factorization.
        info = dicf(n, L)

        # If the factorization exists, then test for termination.
        # Otherwise increment the shift.
        if info >= 0
            # If the shift is at the lower bound, reduce the shift.
            # Otherwise undo the scaling of L and exit.
            if alpha == alphas && nb < nbmax
                alphas /= nbfactor
                alpha = alphas
                nb = nb + 1
            else
                if tx <= n
                    @inbounds for j = 1:n
                        if tx >= j
                            L[tx, j] /= wa2[tx]
                            L[j, tx] = L[tx, j]
                        end
                    end
                end
                CUDA.sync_threads()
                return
            end
        else
            alpha = max(two * alpha, alphas)
        end
    end

    return
end

@inline function dtsol(n::Int, L::CuDeviceArray{Float64,2},
    r::CuDeviceArray{Float64,1})
    # Solve L'*x = r and store the result in r.

    tx = threadIdx().x

    @inbounds for j = n:-1:1
        if tx == 1
            r[j] = r[j] / L[j, j]
        end
        CUDA.sync_threads()

        if tx < j
            r[tx] = r[tx] - L[tx, j] * r[j]
        end
        CUDA.sync_threads()
    end

    return
end

@inline function dmid(n::Int, x::CuDeviceArray{Float64,1},
    xl::CuDeviceArray{Float64,1}, xu::CuDeviceArray{Float64,1})
    tx = threadIdx().x

    if tx <= n
        @inbounds x[tx] = max(xl[tx], min(x[tx], xu[tx]))
    end
    CUDA.sync_threads()

    return
end

@inline function dnrm2(n::Int, x::CuDeviceArray{Float64,1}, incx::Int)
    tx = threadIdx().x

    v = 0.0
    if tx <= n  # No check on ty so that each warp has v.
        @inbounds v = x[tx] * x[tx]
    end

    # shfl_down_sync() will automatically sync threads in a warp.

    offset = 16
    while offset > 0
        v += CUDA.shfl_down_sync(0xffffffff, v, offset)
        offset >>= 1
    end
    v = sqrt(v)
    v = CUDA.shfl_sync(0xffffffff, v, 1)

    return v
end

@inline function dnrm2(n::Int, x::CuDeviceArray{Float64,1}, incx::Int)
    tx = threadIdx().x

    v = 0.0
    if tx <= n  # No check on ty so that each warp has v.
        @inbounds v = x[tx] * x[tx]
    end

    # shfl_down_sync() will automatically sync threads in a warp.

    offset = 16
    while offset > 0
        v += CUDA.shfl_down_sync(0xffffffff, v, offset)
        offset >>= 1
    end
    v = sqrt(v)
    v = CUDA.shfl_sync(0xffffffff, v, 1)

    return v
end

@inline function dnrm2(n::Int, x::CuDeviceArray{Float64,1}, incx::Int)
    tx = threadIdx().x

    v = 0.0
    if tx <= n  # No check on ty so that each warp has v.
        @inbounds v = x[tx] * x[tx]
    end

    # shfl_down_sync() will automatically sync threads in a warp.

    offset = 16
    while offset > 0
        v += CUDA.shfl_down_sync(0xffffffff, v, offset)
        offset >>= 1
    end
    v = sqrt(v)
    v = CUDA.shfl_sync(0xffffffff, v, 1)

    return v
end

@inline function dscal(n::Int, da::Float64, dx::CuDeviceArray{Float64,1}, incx::Int)
    tx = threadIdx().x

    # Ignore incx for now.
    if tx <= n
        @inbounds dx[tx] = da * dx[tx]
    end
    CUDA.sync_threads()

    return
end

@inline function dspcg(n::Int, delta::Float64, rtol::Float64, itermax::Int,
    x::CuDeviceArray{Float64,1}, xl::CuDeviceArray{Float64,1},
    xu::CuDeviceArray{Float64,1}, A::CuDeviceArray{Float64,2},
    g::CuDeviceArray{Float64,1}, s::CuDeviceArray{Float64,1},
    B::CuDeviceArray{Float64,2}, L::CuDeviceArray{Float64,2},
    indfree::CuDeviceArray{Int,1}, gfree::CuDeviceArray{Float64,1},
    w::CuDeviceArray{Float64,1}, iwa::CuDeviceArray{Int,1},
    wa1::CuDeviceArray{Float64,1}, wa2::CuDeviceArray{Float64,1},
    wa3::CuDeviceArray{Float64,1}, wa4::CuDeviceArray{Float64,1},
    wa5::CuDeviceArray{Float64,1})

    tx = threadIdx().x

    zero = 0.0
    one = 1.0

    # Compute A*(x[1] - x[0]) and store in w.

    dssyax(n, A, s, w)

    # Compute the Cauchy point.

    daxpy(n, one, s, 1, x, 1)
    dmid(n, x, xl, xu)

    # Start the main iteration loop.
    # There are at most n iterations because at each iteration
    # at least one variable becomes active.

    info = 3
    iters = 0
    for nfaces = 1:n

        # Determine the free variables at the current minimizer.
        # The indices of the free variables are stored in the first
        # n free positions of the array indfree.
        # The array iwa is used to detect free variables by setting
        # iwa[i] = nfree if the ith variable is free, otherwise iwa[i] = 0.

        # Use a single thread to avoid multiple branch divergences.
        # XXX: Would there be any gain in employing multiple threads?
        nfree = 0
        if tx == 1
            @inbounds for j = 1:n
                if xl[j] < x[j] && x[j] < xu[j]
                    nfree = nfree + 1
                    indfree[nfree] = j
                    iwa[j] = nfree
                else
                    iwa[j] = 0
                end
            end
        end
        nfree = CUDA.shfl_sync(0xffffffff, nfree, 1)

        # Exit if there are no free constraints.

        if nfree == 0
            info = 1
            return info, iters
        end

        # Obtain the submatrix of A for the free variables.
        # Recall that iwa allows the detection of free variables.
        reorder!(n, nfree, B, A, indfree, iwa)

        # Compute the incomplete Cholesky factorization.
        alpha = zero
        dicfs(nfree, alpha, B, L, wa1, wa2)

        # Compute the gradient grad q(x[k]) = g + A*(x[k] - x[0]),
        # of q at x[k] for the free variables.
        # Recall that w contains A*(x[k] - x[0]).
        # Compute the norm of the reduced gradient Z'*g.

        if tx <= nfree
            @inbounds begin
                gfree[tx] = w[indfree[tx]] + g[indfree[tx]]
                wa1[tx] = g[indfree[tx]]
            end
        end
        CUDA.sync_threads()
        gfnorm = dnrm2(nfree, wa1, 1)

        # Save the trust region subproblem in the free variables
        # to generate a direction p[k]. Store p[k] in the array w.

        tol = rtol * gfnorm
        stol = zero

        infotr, itertr = dtrpcg(nfree, B, gfree, delta, L,
            tol, stol, itermax, w,
            wa1, wa2, wa3, wa4, wa5)

        iters += itertr
        dtsol(nfree, L, w)

        # Use a projected search to obtain the next iterate.
        # The projected search algorithm stores s[k] in w.

        if tx <= nfree
            @inbounds begin
                wa1[tx] = x[indfree[tx]]
                wa2[tx] = xl[indfree[tx]]
                wa3[tx] = xu[indfree[tx]]
            end
        end
        CUDA.sync_threads()

        dprsrch(nfree, wa1, wa2, wa3, B, gfree, w, wa4, wa5)

        # Update the minimizer and the step.
        # Note that s now contains x[k+1] - x[0].

        if tx <= nfree
            @inbounds begin
                x[indfree[tx]] = wa1[tx]
                s[indfree[tx]] += w[tx]
            end
        end
        CUDA.sync_threads()

        # Compute A*(x[k+1] - x[0]) and store in w.

        dssyax(n, A, s, w)

        # Compute the gradient grad q(x[k+1]) = g + A*(x[k+1] - x[0])
        # of q at x[k+1] for the free variables.

        if tx == 1
            @inbounds for j = 1:nfree
                gfree[j] = w[indfree[j]] + g[indfree[j]]
            end
        end
        CUDA.sync_threads()

        gfnormf = dnrm2(nfree, gfree, 1)

        # Convergence and termination test.
        # We terminate if the preconditioned conjugate gradient
        # method encounters a direction of negative curvature, or
        # if the step is at the trust region bound.

        if gfnormf <= rtol * gfnorm
            info = 1
            return info, iters
        elseif infotr == 3 || infotr == 4
            info = 2
            return info, iters
        elseif iters > itermax
            info = 3
            return info, iters
        end
    end

    return info, iters
end


@inline function nrm2!(wa, A::CuDeviceArray{Float64,2}, n::Int)
    tx = threadIdx().x

    v = 0.0
    if tx <= n
        @inbounds for j = 1:n
            v += A[j, tx]^2
        end
        @inbounds wa[tx] = sqrt(v)
    end
    #=
    v = A[tx,ty]^2

    if tx > n || ty > n
        v = 0.0
    end

    # Sum over the x-dimension.
    offset = div(blockDim().x, 2)
    while offset > 0
        v += CUDA.shfl_down_sync(0xffffffff, v, offset)
        offset = div(offset, 2)
    end

    if tx == 1
        wa[ty] = sqrt(v)
    end
    =#
    CUDA.sync_threads()

    return
end

@inline function dssyax(n::Int, A::CuDeviceArray{Float64,2},
    z::CuDeviceArray{Float64,1},
    q::CuDeviceArray{Float64,1})
    tx = threadIdx().x

    v = 0.0
    if tx <= n
        @inbounds for j = 1:n
            v += A[tx, j] * z[j]
        end
        @inbounds q[tx] = v
    end
    #=
    v = 0.0
    if tx <= n && ty <= n
        v = A[ty,tx]*z[tx]
    end

    # Sum over the x-dimension: v = sum_tx A[ty,tx]*z[tx].
    # The thread with tx=1 will have the sum in v.

    offset = div(blockDim().x, 2)
    while offset > 0
        v += CUDA.shfl_down_sync(0xffffffff, v, offset)
        offset = div(offset, 2)
    end

    if tx == 1
        q[ty] = v
    end
    =#
    CUDA.sync_threads()

    return
end

@inline function reorder!(n::Int, nfree::Int, B::CuDeviceArray{Float64,2},
    A::CuDeviceArray{Float64,2}, indfree::CuDeviceArray{Int,1},
    iwa::CuDeviceArray{Int,1})
    tx = threadIdx().x

    #=
    if tx == 1 && ty == 1
        @inbounds for j=1:nfree
            jfree = indfree[j]
            B[j,j] = A[jfree,jfree]
            for i=jfree+1:n
                if iwa[i] > 0
                    B[iwa[i],j] = A[i,jfree]
                    B[j,iwa[i]] = B[iwa[i],j]
                end
            end
        end
    end
    =#
    if tx <= nfree
        @inbounds begin
            jfree = indfree[tx]
            B[tx, tx] = A[jfree, jfree]
            for i = jfree+1:n
                if iwa[i] > 0
                    B[iwa[i], tx] = A[i, jfree]
                    B[tx, iwa[i]] = B[iwa[i], tx]
                end
            end
        end
    end

    CUDA.sync_threads()

    return
end

@inline function dtron(n::Int, x::CuDeviceArray{Float64,1}, xl::CuDeviceArray{Float64,1},
    xu::CuDeviceArray{Float64,1}, f::Float64, g::CuDeviceArray{Float64,1},
    A::CuDeviceArray{Float64,2}, frtol::Float64, fatol::Float64,
    fmin::Float64, cgtol::Float64, itermax::Int, delta::Float64, task::Int,
    B::CuDeviceArray{Float64,2}, L::CuDeviceArray{Float64,2},
    xc::CuDeviceArray{Float64,1}, s::CuDeviceArray{Float64,1},
    indfree::CuDeviceArray{Int,1}, gfree::CuDeviceArray{Float64,1},
    isave::CuDeviceArray{Int,1}, dsave::CuDeviceArray{Float64,1},
    wa::CuDeviceArray{Float64,1}, iwa::CuDeviceArray{Int,1},
    wa1::CuDeviceArray{Float64,1}, wa2::CuDeviceArray{Float64,1},
    wa3::CuDeviceArray{Float64,1}, wa4::CuDeviceArray{Float64,1},
    wa5::CuDeviceArray{Float64,1})
    zero = 0.0
    p5 = 0.5
    one = 1.0

    # Parameters for updating the iterates.

    eta0 = 1.0e-4
    eta1 = 0.25
    eta2 = 0.75

    # Parameters for updating the trust region size delta.

    sigma1 = 0.25
    sigma2 = 0.5
    sigma3 = 4.0

    work = 0

    # Initialization section.

    if task == 0  # "START"

        # Initialize local variables.

        iter = 1
        iterscg = 0
        alphac = one
        work = 1  # "COMPUTE"

    else

        @inbounds begin
            # Restore local variables.

            work = isave[1]
            iter = isave[2]
            iterscg = isave[3]
            fc = dsave[1]
            alphac = dsave[2]
            prered = dsave[3]
        end
    end

    CUDA.sync_threads()

    # Search for a lower function value.

    search = true
    while search

        # Compute a step and evaluate the function at the trial point.

        if work == 1 # "COMPUTE"

            # Save the best function value, iterate, and gradient.

            fc = f
            dcopy(n, x, 1, xc, 1)

            # Compute the Cauchy step and store in s.

            alphac = dcauchy(n, x, xl, xu, A, g, delta,
                alphac, s, wa)

            # Compute the projected Newton step.

            info, iters = dspcg(n, delta, cgtol, itermax,
                x, xl, xu, A, g, s,
                B, L,
                indfree, gfree, wa, iwa,
                wa1, wa2, wa3, wa4, wa5)

            # Compute the predicted reduction.

            dssyax(n, A, s, wa)
            prered = -(ddot(n, s, 1, g, 1) + p5 * ddot(n, s, 1, wa, 1))
            iterscg = iterscg + iters

            # Set task to compute the function.

            task = 1 # 'F'
        end

        # Evaluate the step and determine if the step is successful.

        if work == 2 # "EVALUATE"

            # Compute the actual reduction.

            actred = fc - f

            # On the first iteration, adjust the initial step bound.

            snorm = dnrm2(n, s, 1)
            if iter == 1
                delta = min(delta, snorm)
            end

            # Update the trust region bound.

            g0 = ddot(n, g, 1, s, 1)
            if f - fc - g0 <= zero
                alpha = sigma3
            else
                alpha = max(sigma1, -p5 * (g0 / (f - fc - g0)))
            end

            # Update the trust region bound according to the ratio
            # of actual to predicted reduction.

            if actred < eta0 * prered
                delta = min(max(alpha, sigma1) * snorm, sigma2 * delta)
            elseif actred < eta1 * prered
                delta = max(sigma1 * delta, min(alpha * snorm, sigma2 * delta))
            elseif actred < eta2 * prered
                delta = max(sigma1 * delta, min(alpha * snorm, sigma3 * delta))
            else
                delta = max(delta, min(alpha * snorm, sigma3 * delta))
            end

            # Update the iterate.

            if actred > eta0 * prered

                # Successful iterate.

                task = 2 # 'G' or 'H'
                iter = iter + 1

            else

                # Unsuccessful iterate.

                task = 1 # 'F'
                dcopy(n, xc, 1, x, 1)
                f = fc

            end

            # Test for convergence.

            if f < fmin
                task = 10 # "WARNING: F .LT. FMIN"
            end
            if abs(actred) <= fatol && prered <= fatol
                task = 4 # "CONVERGENCE: FATOL TEST SATISFIED"
            end
            if abs(actred) <= frtol * abs(f) && prered <= frtol * abs(f)
                task = 4 # "CONVERGENCE: FRTOL TEST SATISFIED"
            end
        end

        # Test for continuation of search

        if task == 1 && work == 2 # Char(task[1]) == 'F' && work == "EVALUATE"
            search = true
            work = 1 # "COMPUTE"
        else
            search = false
        end
    end

    if work == 3 # "NEWX"
        task = 3 # "NEWX"
    end

    # Decide on what work to perform on the next iteration.

    if task == 1 && work == 1 # Char(task[1]) == 'F' && work == "COMPUTE"
        work = 2 # "EVALUATE"
    elseif task == 1 && work == 2 # Char(task[1]) == 'F' && work == "EVALUATE"
        work = 1 # "COMPUTE"
    elseif task == 2 # unsafe_string(pointer(task),2) == "GH"
        work = 3 # "NEWX"
    elseif task == 3 # unsafe_string(pointer(task),4) == "NEWX"
        work = 1 # "COMPUTE"
    end

    @inbounds begin
        # Save local variables.

        isave[1] = work
        isave[2] = iter
        isave[3] = iterscg

        dsave[1] = fc
        dsave[2] = alphac
        dsave[3] = prered
    end

    CUDA.sync_threads()

    return delta, task
end

@inline function dtrpcg(n::Int, A::CuDeviceArray{Float64,2},
    g::CuDeviceArray{Float64,1}, delta::Float64,
    L::CuDeviceArray{Float64,2},
    tol::Float64, stol::Float64, itermax::Int,
    w::CuDeviceArray{Float64,1},
    p::CuDeviceArray{Float64,1},
    q::CuDeviceArray{Float64,1},
    r::CuDeviceArray{Float64,1},
    t::CuDeviceArray{Float64,1},
    z::CuDeviceArray{Float64,1})
    zero = 0.0
    one = 1.0

    # Initialize the iterate w and the residual r.
    fill!(w, 0)

    # Initialize the residual t of grad q to -g.
    # Initialize the residual r of grad Q by solving L*r = -g.
    # Note that t = L*r.
    dcopy(n, g, 1, t, 1)
    dscal(n, -one, t, 1)
    dcopy(n, t, 1, r, 1)
    dnsol(n, L, r)

    # Initialize the direction p.
    dcopy(n, r, 1, p, 1)

    # Initialize rho and the norms of r and t.
    rho = ddot(n, r, 1, r, 1)
    rnorm0 = sqrt(rho)

    # Exit if g = 0.
    iters = 0
    if rnorm0 == zero
        iters = 0
        info = 1
        return info, iters
    end

    for iters = 1:itermax

        # Note:
        # Q(w) = 0.5*w'Bw + h'w, where B=L^{-1}AL^{-T}, h=L^{-1}g.
        # Then p'Bp = p'L^{-1}AL^{-T}p = p'L^{-1}Az = p'q.
        # alpha = r'r / p'Bp.

        dcopy(n, p, 1, z, 1)
        dtsol(n, L, z)

        # Compute q by solving L*q = A*z and save L*q for
        # use in updating the residual t.
        dssyax(n, A, z, q)
        dcopy(n, q, 1, z, 1)
        dnsol(n, L, q)

        # Compute alpha and determine sigma such that the trust region
        # constraint || w + sigma*p || = delta is satisfied.
        ptq = ddot(n, p, 1, q, 1)
        if ptq > zero
            alpha = rho / ptq
        else
            alpha = zero
        end
        sigma = dtrqsol(n, w, p, delta)

        # Exit if there is negative curvature or if the
        # iterates exit the trust region.

        if (ptq <= zero) || (alpha >= sigma)
            daxpy(n, sigma, p, 1, w, 1)
            if ptq <= zero
                info = 3
            else
                info = 4
            end

            return info, iters
        end

        # Update w and the residuals r and t.
        # Note that t = L*r.

        daxpy(n, alpha, p, 1, w, 1)
        daxpy(n, -alpha, q, 1, r, 1)
        daxpy(n, -alpha, z, 1, t, 1)

        # Exit if the residual convergence test is satisfied.

        rtr = ddot(n, r, 1, r, 1)
        rnorm = sqrt(rtr)
        tnorm = sqrt(ddot(n, t, 1, t, 1))

        if tnorm <= tol
            info = 1
            return info, iters
        end

        if rnorm <= stol
            info = 2
            return info, iters
        end

        # Compute p = r + beta*p and update rho.
        beta = rtr / rho
        dscal(n, beta, p, 1)
        daxpy(n, one, r, 1, p, 1)
        rho = rtr
    end

    iters = itermax
    info = 5
    return info, iters
end

@inline function dtrqsol(n::Int, x::CuDeviceArray{Float64,1},
    p::CuDeviceArray{Float64,1}, delta::Float64)
    zero = 0.0
    sigma = zero

    ptx = ddot(n, p, 1, x, 1)
    ptp = ddot(n, p, 1, p, 1)
    xtx = ddot(n, x, 1, x, 1)
    dsq = delta^2

    # Guard against abnormal cases.
    rad = ptx^2 + ptp * (dsq - xtx)
    rad = sqrt(max(rad, zero))

    if ptx > zero
        sigma = (dsq - xtx) / (ptx + rad)
    elseif rad > zero
        sigma = (rad - ptx) / ptp
    else
        sigma = zero
    end
    CUDA.sync_threads()

    return sigma
end

@testset "dicf" begin
    function dicf_test(d_in::CuDeviceArray{Float64},
        d_out::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        L = CuDynamicSharedArray(Float64, (n, n))
        for i in 1:n
            L[i, tx] = d_in[i, tx]
        end
        CUDA.sync_threads()

        # Test Cholesky factorization.
        dicf(n, L)

        if bx == 1
            for i in 1:n
                d_out[i, tx] = L[i, tx]
            end
        end
        CUDA.sync_threads()
        return
    end

    for i = 1:itermax
        L = tril(rand(n, n))
        A = L * transpose(L)
        A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
        tron_A = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
        tron_A.vals .= A

        d_in = CuArray{Float64,2}(undef, (n, n))
        d_out = CuArray{Float64,2}(undef, (n, n))
        copyto!(d_in, tron_A.vals)
        b = @benchmark (CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ($n^2 * sizeof(Float64)) $dicf_test($d_in, $d_out))
        display(b)
        save_benchmark(b, "dicf.json")
        h_L = zeros(n, n)
        copyto!(h_L, d_out)

        tron_L = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
        tron_L.vals .= tron_A.vals
        indr = zeros(Int, n)
        indf = zeros(n)
        list = zeros(n)
        w = zeros(n)
        ExaTronKernels.dicf(n, n^2, tron_L, 5, indr, indf, list, w)

        @test norm(tron_A.vals .- tril(h_L) * transpose(tril(h_L))) <= 1e-10
        @test norm(tril(h_L) .- transpose(triu(h_L))) <= 1e-10
        @test norm(tril(tron_L.vals) .- tril(h_L)) <= 1e-10
    end
end

@testset "dicfs" begin
    function dicfs_test(alpha::Float64,
        dA::CuDeviceArray{Float64},
        d_out::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        wa1 = CuDynamicSharedArray(Float64, n)
        wa2 = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))
        A = CuDynamicSharedArray(Float64, (n, n), (2 * n) * sizeof(Float64))
        L = CuDynamicSharedArray(Float64, (n, n), (2 * n + n^2) * sizeof(Float64))

        @inbounds for j = 1:n
            A[j, tx] = dA[j, tx]
        end
        CUDA.sync_threads()

        dicfs(n, alpha, A, L, wa1, wa2)
        if bx == 1
            @inbounds for j = 1:n
                d_out[j, tx] = L[j, tx]
            end
        end
        CUDA.sync_threads()

        return
    end

    for i = 1:itermax
        L = tril(rand(n, n))
        A = L * transpose(L)
        A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
        tron_A = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
        tron_L = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
        tron_A.vals .= A

        dA = CuArray{Float64,2}(undef, (n, n))
        d_out = CuArray{Float64,2}(undef, (n, n))
        alpha = 1.0
        copyto!(dA, tron_A.vals)
        b = @benchmark (CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ((2 * $n + 2 * $n^2) * sizeof(Float64)) $dicfs_test($alpha, $dA, $d_out))
        display(b)
        save_benchmark(b, "dicfs.json")
        h_L = zeros(n, n)
        copyto!(h_L, d_out)
        iwa = zeros(Int, 3 * n)
        wa1 = zeros(n)
        wa2 = zeros(n)
        ExaTronKernels.dicfs(n, n^2, tron_A, tron_L, 5, alpha, iwa, wa1, wa2)

        @test norm(tril(h_L) .- transpose(triu(h_L))) <= 1e-10
        @test norm(tril(tron_L.vals) .- tril(h_L)) <= 1e-9

        # Make it negative definite.
        for j = 1:n
            tron_A.vals[j, j] = -tron_A.vals[j, j]
        end
        copyto!(dA, tron_A.vals)
        b = @benchmark (CUDA.@sync @cuda threads = n blocks = $nblk shmem = ((2 * $n + 2 * $n^2) * sizeof(Float64)) $dicfs_test($alpha, $dA, $d_out))
        copyto!(h_L, d_out)
        ExaTronKernels.dicfs(n, n^2, tron_A, tron_L, 5, alpha, iwa, wa1, wa2)

        @test norm(tril(h_L) .- transpose(triu(h_L))) <= 1e-10
        @test norm(tril(tron_L.vals) .- tril(h_L)) <= 1e-10
    end
end

@testset "dcauchy" begin
    function dcauchy_test(dx::CuDeviceArray{Float64},
        dl::CuDeviceArray{Float64},
        du::CuDeviceArray{Float64},
        dA::CuDeviceArray{Float64},
        dg::CuDeviceArray{Float64},
        delta::Float64,
        alpha::Float64,
        d_out1::CuDeviceArray{Float64},
        d_out2::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        x = CuDynamicSharedArray(Float64, n)
        xl = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))
        xu = CuDynamicSharedArray(Float64, n, (2 * n) * sizeof(Float64))
        g = CuDynamicSharedArray(Float64, n, (3 * n) * sizeof(Float64))
        s = CuDynamicSharedArray(Float64, n, (4 * n) * sizeof(Float64))
        wa = CuDynamicSharedArray(Float64, n, (5 * n) * sizeof(Float64))
        A = CuDynamicSharedArray(Float64, (n, n), (6 * n) * sizeof(Float64))

        for i in 1:n
            A[i, tx] = dA[i, tx]
        end
        x[tx] = dx[tx]
        xl[tx] = dl[tx]
        xu[tx] = du[tx]
        g[tx] = dg[tx]

        alpha = dcauchy(n, x, xl, xu, A, g, delta, alpha, s, wa)
        if bx == 1
            d_out1[tx] = s[tx]
            d_out2[tx] = alpha
        end
        CUDA.sync_threads()

        return
    end

    for i = 1:itermax
        L = tril(rand(n, n))
        A = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
        A.vals .= L * transpose(L)
        A.vals .= tril(A.vals) .+ (transpose(tril(A.vals)) .- Diagonal(A.vals))
        x = rand(n)
        xl = x .- abs.(rand(n))
        xu = x .+ abs.(rand(n))
        g = A.vals * x .+ rand(n)
        s = zeros(n)
        wa = zeros(n)
        alpha = 1.0
        delta = 2.0 * norm(g)

        dx = CuArray{Float64}(undef, n)
        dl = CuArray{Float64}(undef, n)
        du = CuArray{Float64}(undef, n)
        dg = CuArray{Float64}(undef, n)
        dA = CuArray{Float64,2}(undef, (n, n))
        d_out1 = CuArray{Float64}(undef, n)
        d_out2 = CuArray{Float64}(undef, n)
        copyto!(dx, x)
        copyto!(dl, xl)
        copyto!(du, xu)
        copyto!(dg, g)
        copyto!(dA, A.vals)
        b = @benchmark (CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ((6 * $n + $n^2) * sizeof(Float64)) $dcauchy_test($dx, $dl, $du, $dA, $dg, $delta, $alpha, $d_out1, $d_out2))
        display(b)
        save_benchmark(b, "dcauchy.json")
        h_s = zeros(n)
        h_alpha = zeros(n)
        copyto!(h_s, d_out1)
        copyto!(h_alpha, d_out2)

        alpha = ExaTronKernels.dcauchy(n, x, xl, xu, A, g, delta, alpha, s, wa)

        @test norm(s .- h_s) <= 1e-10
        @test norm(alpha .- h_alpha) <= 1e-10
    end
end

@testset "dtrpcg" begin
    function dtrpcg_test(delta::Float64, tol::Float64,
        stol::Float64, d_in::CuDeviceArray{Float64},
        d_g::CuDeviceArray{Float64},
        d_out_L::CuDeviceArray{Float64},
        d_out::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        A = CuDynamicSharedArray(Float64, (n, n))
        L = CuDynamicSharedArray(Float64, (n, n), (n^2) * sizeof(Float64))

        g = CuDynamicSharedArray(Float64, n, (2 * n^2) * sizeof(Float64))
        w = CuDynamicSharedArray(Float64, n, (2 * n^2 + n) * sizeof(Float64))
        p = CuDynamicSharedArray(Float64, n, (2 * n^2 + 2 * n) * sizeof(Float64))
        q = CuDynamicSharedArray(Float64, n, (2 * n^2 + 3 * n) * sizeof(Float64))
        r = CuDynamicSharedArray(Float64, n, (2 * n^2 + 4 * n) * sizeof(Float64))
        t = CuDynamicSharedArray(Float64, n, (2 * n^2 + 5 * n) * sizeof(Float64))
        z = CuDynamicSharedArray(Float64, n, (2 * n^2 + 6 * n) * sizeof(Float64))

        for i in 1:n
            A[i, tx] = d_in[i, tx]
            L[i, tx] = d_in[i, tx]
        end
        g[tx] = d_g[tx]
        CUDA.sync_threads()

        dicf(n, L)
        info, iters = dtrpcg(n, A, g, delta, L, tol, stol, n, w, p, q, r, t, z)
        if bx == 1
            d_out[tx] = w[tx]
            for i in 1:n
                d_out_L[i, tx] = L[i, tx]
            end
        end
        CUDA.sync_threads()

        return
    end

    delta = 100.0
    tol = 1e-6
    stol = 1e-6
    tron_A = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
    tron_L = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
    for i = 1:itermax
        L = tril(rand(n, n))
        A = L * transpose(L)
        A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
        g = 0.1 * ones(n)
        w = zeros(n)
        p = zeros(n)
        q = zeros(n)
        r = zeros(n)
        t = zeros(n)
        z = zeros(n)
        tron_A.vals .= A
        tron_L.vals .= A
        d_in = CuArray{Float64,2}(undef, (n, n))
        d_g = CuArray{Float64}(undef, n)
        d_out_L = CuArray{Float64,2}(undef, (n, n))
        d_out = CuArray{Float64}(undef, n)
        copyto!(d_in, A)
        copyto!(d_g, g)
        b = @benchmark(CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ((2 * $n^2 + 7 * $n) * sizeof(Float64)) $dtrpcg_test($delta, $tol, $stol, $d_in, $d_g, $d_out_L, $d_out))
        display(b)
        save_benchmark(b, "dtrpcg.json")
        h_w = zeros(n)
        h_L = zeros(n, n)
        copyto!(h_L, d_out_L)
        copyto!(h_w, d_out)

        indr = zeros(Int, n)
        indf = zeros(n)
        list = zeros(n)
        ExaTronKernels.dicf(n, n^2, tron_L, 5, indr, indf, list, w)
        ExaTronKernels.dtrpcg(n, tron_A, g, delta, tron_L, tol, stol, n, w, p, q, r, t, z)

        @test norm(tril(h_L) .- tril(tron_L.vals)) <= tol
        @test norm(h_w .- w) <= tol
    end
end

@testset "dprsrch" begin
    function dprsrch_test(d_x::CuDeviceArray{Float64},
        d_xl::CuDeviceArray{Float64},
        d_xu::CuDeviceArray{Float64},
        d_g::CuDeviceArray{Float64},
        d_w::CuDeviceArray{Float64},
        d_A::CuDeviceArray{Float64},
        d_out1::CuDeviceArray{Float64},
        d_out2::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        x = CuDynamicSharedArray(Float64, n)
        xl = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))
        xu = CuDynamicSharedArray(Float64, n, (2 * n) * sizeof(Float64))
        g = CuDynamicSharedArray(Float64, n, (3 * n) * sizeof(Float64))
        w = CuDynamicSharedArray(Float64, n, (4 * n) * sizeof(Float64))
        wa1 = CuDynamicSharedArray(Float64, n, (5 * n) * sizeof(Float64))
        wa2 = CuDynamicSharedArray(Float64, n, (6 * n) * sizeof(Float64))
        A = CuDynamicSharedArray(Float64, (n, n), (7 * n) * sizeof(Float64))
        for i in 1:n
            A[i, tx] = d_A[i, tx]
        end
        x[tx] = d_x[tx]
        xl[tx] = d_xl[tx]
        xu[tx] = d_xu[tx]
        g[tx] = d_g[tx]
        w[tx] = d_w[tx]
        CUDA.sync_threads()

        dprsrch(n, x, xl, xu, A, g, w, wa1, wa2)
        if bx == 1
            d_out1[tx] = x[tx]
            d_out2[tx] = w[tx]
        end
        CUDA.sync_threads()

        return
    end

    for i = 1:itermax
        L = tril(rand(n, n))
        A = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
        A.vals .= L * transpose(L)
        A.vals .= tril(A.vals) .+ (transpose(tril(A.vals)) .- Diagonal(A.vals))
        x = rand(n)
        xl = x .- abs.(rand(n))
        xu = x .+ abs.(rand(n))
        g = A.vals * x .+ rand(n)
        w = -g
        wa1 = zeros(n)
        wa2 = zeros(n)

        dx = CuArray{Float64}(undef, n)
        dl = CuArray{Float64}(undef, n)
        du = CuArray{Float64}(undef, n)
        dg = CuArray{Float64}(undef, n)
        dw = CuArray{Float64}(undef, n)
        dA = CuArray{Float64,2}(undef, (n, n))
        d_out1 = CuArray{Float64}(undef, n)
        d_out2 = CuArray{Float64}(undef, n)
        copyto!(dx, x)
        copyto!(dl, xl)
        copyto!(du, xu)
        copyto!(dg, g)
        copyto!(dw, w)
        copyto!(dA, A.vals)
        b = @benchmark(CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ((7 * $n + $n^2) * sizeof(Float64)) $dprsrch_test($dx, $dl, $du, $dg, $dw, $dA, $d_out1, $d_out2))
        display(b)
        save_benchmark(b, "dprsrch.json")
        h_x = zeros(n)
        h_w = zeros(n)
        copyto!(h_x, d_out1)
        copyto!(h_w, d_out2)

        ExaTronKernels.dprsrch(n, x, xl, xu, A, g, w, wa1, wa2)

        @test norm(x .- h_x) <= 1e-10
        @test norm(w .- h_w) <= 1e-10
    end
end

@testset "daxpy" begin
    function daxpy_test(da, d_in::CuDeviceArray{Float64},
        d_out::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        x = CuDynamicSharedArray(Float64, n)
        y = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))
        x[tx] = d_in[tx]
        y[tx] = d_in[tx+n]
        CUDA.sync_threads()

        daxpy(n, da, x, 1, y, 1)
        if bx == 1
            d_out[tx] = y[tx]
        end
        CUDA.sync_threads()

        return
    end

    for i = 1:itermax
        da = rand(1)[1]
        h_in = rand(2 * n)
        h_out = zeros(n)
        d_in = CuArray{Float64}(undef, 2 * n)
        d_out = CuArray{Float64}(undef, n)
        copyto!(d_in, h_in)
        b = @benchmark (CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ((2 * $n) * sizeof(Float64)) $daxpy_test($da, $d_in, $d_out))
        display(b)
        save_benchmark(b, "daxpy.json")
        copyto!(h_out, d_out)

        @test norm(h_out .- (h_in[n+1:2*n] .+ da .* h_in[1:n])) <= 1e-12
    end
end
@testset "dssyax" begin
    function dssyax_test(d_z::CuDeviceArray{Float64},
        d_in::CuDeviceArray{Float64},
        d_out::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        z = CuDynamicSharedArray(Float64, n)
        q = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))
        A = CuDynamicSharedArray(Float64, (n, n), (2 * n) * sizeof(Float64))
        for i in 1:n
            A[i, tx] = d_in[i, tx]
        end
        z[tx] = d_z[tx]
        CUDA.sync_threads()

        dssyax(n, A, z, q)
        if bx == 1
            d_out[tx] = q[tx]
        end
        CUDA.sync_threads()

        return
    end

    for i = 1:itermax
        z = rand(n)
        h_in = rand(n, n)
        h_out = zeros(n)
        d_z = CuArray{Float64}(undef, n)
        d_in = CuArray{Float64,2}(undef, (n, n))
        d_out = CuArray{Float64}(undef, n)
        copyto!(d_z, z)
        copyto!(d_in, h_in)
        b = @benchmark(CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ((2 * $n + $n^2) * sizeof(Float64)) $dssyax_test($d_z, $d_in, $d_out))
        display(b)
        save_benchmark(b, "dssyax.json")
        copyto!(h_out, d_out)

        @test norm(h_out .- h_in * z) <= 1e-12
    end
end
@testset "dmid" begin
    function dmid_test(dx::CuDeviceArray{Float64},
        dl::CuDeviceArray{Float64},
        du::CuDeviceArray{Float64},
        d_out::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        x = CuDynamicSharedArray(Float64, n)
        xl = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))
        xu = CuDynamicSharedArray(Float64, n, (2 * n) * sizeof(Float64))
        x[tx] = dx[tx]
        xl[tx] = dl[tx]
        xu[tx] = du[tx]
        CUDA.sync_threads()

        dmid(n, x, xl, xu)
        if bx == 1
            d_out[tx] = x[tx]
        end
        CUDA.sync_threads()

        return
    end

    for i = 1:itermax
        x = rand(n)
        xl = x .- abs.(rand(n))
        xu = x .+ abs.(rand(n))

        # Force some components to go below or above bounds
        # so that we can test all cases.
        for j = 1:n
            k = rand(1:3)
            if k == 1
                x[j] = xl[j] - 0.1
            elseif k == 2
                x[j] = xu[j] + 0.1
            end
        end
        x_out = zeros(n)
        dx = CuArray{Float64}(undef, n)
        dl = CuArray{Float64}(undef, n)
        du = CuArray{Float64}(undef, n)
        d_out = CuArray{Float64}(undef, n)
        copyto!(dx, x)
        copyto!(dl, xl)
        copyto!(du, xu)

        b = @benchmark(CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ((3 * $n) * sizeof(Float64)) $dmid_test($dx, $dl, $du, $d_out))
        display(b)
        save_benchmark(b, "dmid.json")
        copyto!(x_out, d_out)

        ExaTronKernels.dmid(n, x, xl, xu)
        @test !(false in (x .== x_out))
    end
end


@testset "dgpstep" begin
    function dgpstep_test(dx::CuDeviceArray{Float64},
        dl::CuDeviceArray{Float64},
        du::CuDeviceArray{Float64},
        alpha::Float64,
        dw::CuDeviceArray{Float64},
        d_out::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        x = CuDynamicSharedArray(Float64, n)
        xl = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))
        xu = CuDynamicSharedArray(Float64, n, (2 * n) * sizeof(Float64))
        w = CuDynamicSharedArray(Float64, n, (3 * n) * sizeof(Float64))
        s = CuDynamicSharedArray(Float64, n, (4 * n) * sizeof(Float64))
        x[tx] = dx[tx]
        xl[tx] = dl[tx]
        xu[tx] = du[tx]
        w[tx] = dw[tx]
        CUDA.sync_threads()

        dgpstep(n, x, xl, xu, alpha, w, s)
        if bx == 1
            d_out[tx] = s[tx]
        end
        CUDA.sync_threads()

        return
    end

    for i = 1:itermax
        x = rand(n)
        xl = x .- abs.(rand(n))
        xu = x .+ abs.(rand(n))
        w = rand(n)
        alpha = rand(1)[1]
        s = zeros(n)
        s_out = zeros(n)

        # Force some components to go below or above bounds
        # so that we can test all cases.
        for j = 1:n
            k = rand(1:3)
            if k == 1
                if x[j] + alpha * w[j] >= xl[j]
                    w[j] = (xl[j] - x[j]) / alpha - 0.1
                end
            elseif k == 2
                if x[j] + alpha * w[j] <= xu[j]
                    w[j] = (xu[j] - x[j]) / alpha + 0.1
                end
            end
        end

        dx = CuArray{Float64}(undef, n)
        dl = CuArray{Float64}(undef, n)
        du = CuArray{Float64}(undef, n)
        dw = CuArray{Float64}(undef, n)
        d_out = CuArray{Float64}(undef, n)
        copyto!(dx, x)
        copyto!(dl, xl)
        copyto!(du, xu)
        copyto!(dw, w)
        b = @benchmark(CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ((5 * $n) * sizeof(Float64)) $dgpstep_test($dx, $dl, $du, $alpha, $dw, $d_out))
        display(b)
        save_benchmark(b, "dgpstep.json")
        copyto!(s_out, d_out)

        ExaTronKernels.dgpstep(n, x, xl, xu, alpha, w, s)
        @test !(false in (s .== s_out))
    end
end

@testset "dbreakpt" begin
    function dbreakpt_test(dx::CuDeviceArray{Float64},
        dl::CuDeviceArray{Float64},
        du::CuDeviceArray{Float64},
        dw::CuDeviceArray{Float64},
        d_nbrpt::CuDeviceArray{Float64},
        d_brptmin::CuDeviceArray{Float64},
        d_brptmax::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        x = CuDynamicSharedArray(Float64, n)
        xl = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))
        xu = CuDynamicSharedArray(Float64, n, (2 * n) * sizeof(Float64))
        w = CuDynamicSharedArray(Float64, n, (3 * n) * sizeof(Float64))
        x[tx] = dx[tx]
        xl[tx] = dl[tx]
        xu[tx] = du[tx]
        w[tx] = dw[tx]
        CUDA.sync_threads()

        nbrpt, brptmin, brptmax = dbreakpt(n, x, xl, xu, w)
        for i in 1:n
            d_nbrpt[i, tx] = nbrpt
            d_brptmin[i, tx] = brptmin
            d_brptmax[i, tx] = brptmax
        end
        CUDA.sync_threads()

        return
    end

    for i = 1:itermax
        x = rand(n)
        xl = x .- abs.(rand(n))
        xu = x .+ abs.(rand(n))
        w = 2.0 * rand(n) .- 1.0     # (-1,1]
        h_nbrpt = zeros((n, n))
        h_brptmin = zeros((n, n))
        h_brptmax = zeros((n, n))

        dx = CuArray{Float64}(undef, n)
        dl = CuArray{Float64}(undef, n)
        du = CuArray{Float64}(undef, n)
        dw = CuArray{Float64}(undef, n)
        d_nbrpt = CuArray{Float64,2}(undef, (n, n))
        d_brptmin = CuArray{Float64,2}(undef, (n, n))
        d_brptmax = CuArray{Float64,2}(undef, (n, n))
        copyto!(dx, x)
        copyto!(dl, xl)
        copyto!(du, xu)
        copyto!(dw, w)
        b = @benchmark(CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ((4 * $n) * sizeof(Float64)) $dbreakpt_test($dx, $dl, $du, $dw, $d_nbrpt, $d_brptmin, $d_brptmax))
        display(b)
        save_benchmark(b, "dbreakpt.json")
        copyto!(h_nbrpt, d_nbrpt)
        copyto!(h_brptmin, d_brptmin)
        copyto!(h_brptmax, d_brptmax)

        nbrpt, brptmin, brptmax = ExaTronKernels.dbreakpt(n, x, xl, xu, w)
        @test !(false in (nbrpt .== h_nbrpt))
        @test !(false in (brptmin .== h_brptmin))
        @test !(false in (brptmax .== h_brptmax))
    end
end

@testset "dnrm2" begin
    function dnrm2_test(d_in::CuDeviceArray{Float64},
        d_out::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        x = CuDynamicSharedArray(Float64, n)
        x[tx] = d_in[tx]
        CUDA.sync_threads()

        v = dnrm2(n, x, 1)
        if bx == 1
            d_out[tx] = v
        end
        CUDA.sync_threads()

        return
    end

    @inbounds for i = 1:itermax
        h_in = rand(n)
        h_out = zeros(n)
        d_in = CuArray{Float64}(undef, n)
        d_out = CuArray{Float64}(undef, n)
        copyto!(d_in, h_in)
        b = @benchmark(CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ($n * sizeof(Float64)) $dnrm2_test($d_in, $d_out))
        display(b)
        save_benchmark(b, "dnrm2.json")
        copyto!(h_out, d_out)
        xnorm = ExaTronKernels.norm(h_in, 2)

        @test norm(xnorm .- h_out) <= 1e-10
    end
end

@testset "nrm2" begin
    function nrm2_test(d_A::CuDeviceArray{Float64}, d_out::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        wa = CuDynamicSharedArray(Float64, n)
        A = CuDynamicSharedArray(Float64, (n, n), n * sizeof(Float64))

        for i in 1:n
            A[i, tx] = d_A[i, tx]
        end
        CUDA.sync_threads()

        nrm2!(wa, A, n)
        if bx == 1
            d_out[tx] = wa[tx]
        end
        CUDA.sync_threads()

        return
    end

    @inbounds for i = 1:itermax
        L = tril(rand(n, n))
        A = L * transpose(L)
        A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
        wa = zeros(n)
        tron_A = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
        tron_A.vals .= A
        ExaTronKernels.nrm2!(wa, tron_A, n)

        d_A = CuArray{Float64,2}(undef, (n, n))
        d_out = CuArray{Float64}(undef, n)
        h_wa = zeros(n)
        copyto!(d_A, A)
        b = @benchmark(CUDA.@sync @cuda threads = ($n, $n) blocks = $nblk shmem = (($n^2 + $n) * sizeof(Float64)) $nrm2_test($d_A, $d_out))
        display(b)
        save_benchmark(b, "nrm2.json")
        copyto!(h_wa, d_out)

        @test norm(wa .- h_wa) <= 1e-10
    end
end


@testset "dcopy" begin
    function dcopy_test(d_in::CuDeviceArray{Float64},
        d_out::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        x = CuDynamicSharedArray(Float64, n)
        y = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))

        x[tx] = d_in[tx]
        CUDA.sync_threads()

        dcopy(n, x, 1, y, 1)

        if bx == 1
            d_out[tx] = y[tx]
        end
        CUDA.sync_threads()

        return
    end

    @inbounds for i = 1:itermax
        h_in = rand(n)
        h_out = zeros(n)
        d_in = CuArray{Float64}(undef, n)
        d_out = CuArray{Float64}(undef, n)
        copyto!(d_in, h_in)
        b = @benchmark(CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ((2 * $n) * sizeof(Float64)) $dcopy_test($d_in, $d_out))
        display(b)
        save_benchmark(b, "dcopy.json")
        copyto!(h_out, d_out)

        @test !(false in (h_in .== h_out))
    end
end

@testset "ddot" begin
    function ddot_test(d_in::CuDeviceArray{Float64},
        d_out::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        x = CuDynamicSharedArray(Float64, n)
        y = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))
        x[tx] = d_in[tx]
        y[tx] = d_in[tx]
        CUDA.sync_threads()

        v = ddot(n, x, 1, y, 1)
        if bx == 1
            for i in 1:n
                d_out[i, tx] = v
            end
        end
        CUDA.sync_threads()

        return
    end

    @inbounds for i = 1:itermax
        h_in = rand(n)
        h_out = zeros((n, n))
        d_in = CuArray{Float64}(undef, n)
        d_out = CuArray{Float64,2}(undef, (n, n))
        copyto!(d_in, h_in)
        b = @benchmark(CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ((2 * $n) * sizeof(Float64)) $ddot_test($d_in, $d_out))
        display(b)
        save_benchmark(b, "ddot.json")
        copyto!(h_out, d_out)

        @test norm(dot(h_in, h_in) .- h_out, 2) <= 1e-10
    end
end



@testset "dscal" begin
    function dscal_test(da::Float64,
        d_in::CuDeviceArray{Float64},
        d_out::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        x = CuDynamicSharedArray(Float64, n)
        x[tx] = d_in[tx]
        CUDA.sync_threads()

        dscal(n, da, x, 1)
        if bx == 1
            d_out[tx] = x[tx]
        end
        CUDA.sync_threads()

        return
    end

    for i = 1:itermax
        h_in = rand(n)
        h_out = zeros(n)
        da = rand(1)[1]
        d_in = CuArray{Float64}(undef, n)
        d_out = CuArray{Float64}(undef, n)
        copyto!(d_in, h_in)
        b = @benchmark(CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ($n * sizeof(Float64)) $dscal_test($da, $d_in, $d_out))
        display(b)
        save_benchmark(b, "dscal.json")
        copyto!(h_out, d_out)

        @test norm(h_out .- (da .* h_in)) <= 1e-12
    end
end

@testset "dtrqsol" begin
    function dtrqsol_test(d_x::CuDeviceArray{Float64},
        d_p::CuDeviceArray{Float64},
        d_out::CuDeviceArray{Float64},
        delta::Float64)
        tx = CUDA.threadIdx().x

        x = CuDynamicSharedArray(Float64, n)
        p = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))

        x[tx] = d_x[tx]
        p[tx] = d_p[tx]
        CUDA.sync_threads()

        sigma = dtrqsol(n, x, p, delta)
        for i in 1:n
            d_out[i, tx] = sigma
        end
        CUDA.sync_threads()
    end

    for i = 1:itermax
        x = rand(n)
        p = rand(n)
        sigma = abs(rand(1)[1])
        delta = norm(x .+ sigma .* p)

        d_x = CuArray{Float64}(undef, n)
        d_p = CuArray{Float64}(undef, n)
        d_out = CuArray{Float64,2}(undef, (n, n))
        copyto!(d_x, x)
        copyto!(d_p, p)
        b = @benchmark (CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ((2 * $n) * sizeof(Float64)) $dtrqsol_test($d_x, $d_p, $d_out, $delta))
        display(b)
        save_benchmark(b, "dtrqsol.json")
        @test norm(sigma .- d_out) <= 1e-10
    end
end

@testset "dspcg" begin
    function dspcg_test(delta::Float64, rtol::Float64,
        cg_itermax::Int, dx::CuDeviceArray{Float64},
        dxl::CuDeviceArray{Float64},
        dxu::CuDeviceArray{Float64},
        dA::CuDeviceArray{Float64},
        dg::CuDeviceArray{Float64},
        ds::CuDeviceArray{Float64},
        d_out::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        x = CuDynamicSharedArray(Float64, n)
        xl = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))
        xu = CuDynamicSharedArray(Float64, n, (2 * n) * sizeof(Float64))
        g = CuDynamicSharedArray(Float64, n, (3 * n) * sizeof(Float64))
        s = CuDynamicSharedArray(Float64, n, (4 * n) * sizeof(Float64))
        w = CuDynamicSharedArray(Float64, n, (5 * n) * sizeof(Float64))
        wa1 = CuDynamicSharedArray(Float64, n, (6 * n) * sizeof(Float64))
        wa2 = CuDynamicSharedArray(Float64, n, (7 * n) * sizeof(Float64))
        wa3 = CuDynamicSharedArray(Float64, n, (8 * n) * sizeof(Float64))
        wa4 = CuDynamicSharedArray(Float64, n, (9 * n) * sizeof(Float64))
        wa5 = CuDynamicSharedArray(Float64, n, (10 * n) * sizeof(Float64))
        gfree = CuDynamicSharedArray(Float64, n, (11 * n) * sizeof(Float64))
        indfree = CuDynamicSharedArray(Int, n, (12 * n) * sizeof(Float64))
        iwa = CuDynamicSharedArray(Int, 2 * n, n * sizeof(Int) + (12 * n) * sizeof(Float64))

        A = CuDynamicSharedArray(Float64, (n, n), (12 * n) * sizeof(Float64) + (3 * n) * sizeof(Int))
        B = CuDynamicSharedArray(Float64, (n, n), (12 * n + n^2) * sizeof(Float64) + (3 * n) * sizeof(Int))
        L = CuDynamicSharedArray(Float64, (n, n), (12 * n + 2 * n^2) * sizeof(Float64) + (3 * n) * sizeof(Int))

        @inbounds for j = 1:n
            A[j, tx] = dA[j, tx]
        end
        x[tx] = dx[tx]
        xl[tx] = dxl[tx]
        xu[tx] = dxu[tx]
        g[tx] = dg[tx]
        s[tx] = ds[tx]
        CUDA.sync_threads()

        dspcg(n, delta, rtol, cg_itermax, x, xl, xu,
            A, g, s, B, L, indfree, gfree, w, iwa,
            wa1, wa2, wa3, wa4, wa5)

        if bx == 1
            d_out[tx] = x[tx]
        end
        CUDA.sync_threads()

        return
    end

    for i = 1:itermax
        L = tril(rand(n, n))
        A = L * transpose(L)
        A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
        tron_A = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
        tron_A.vals .= A
        tron_B = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
        tron_L = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
        x = rand(n)
        xl = x .- abs.(rand(n))
        xu = x .+ abs.(rand(n))
        g = A * x .+ rand(n)
        s = rand(n)
        delta = 2.0 * norm(g)
        rtol = 1e-6
        cg_itermax = n
        w = zeros(n)
        wa = zeros(5 * n)
        gfree = zeros(n)
        indfree = zeros(Int, n)
        iwa = zeros(Int, 3 * n)

        dx = CuArray{Float64}(undef, n)
        dxl = CuArray{Float64}(undef, n)
        dxu = CuArray{Float64}(undef, n)
        dA = CuArray{Float64,2}(undef, (n, n))
        dg = CuArray{Float64}(undef, n)
        ds = CuArray{Float64}(undef, n)
        d_out = CuArray{Float64}(undef, n)

        copyto!(dx, x)
        copyto!(dxl, xl)
        copyto!(dxu, xu)
        copyto!(dA, tron_A.vals)
        copyto!(dg, g)
        copyto!(ds, s)

        b = @benchmark(CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ((3 * $n) * sizeof(Int) + (12 * $n + 3 * ($n^2)) * sizeof(Float64)) $dspcg_test($delta, $rtol, $cg_itermax, $dx, $dxl, $dxu, $dA, $dg, $ds, $d_out))
        display(b)
        save_benchmark(b, "dspcg.json")
        h_x = zeros(n)
        copyto!(h_x, d_out)

        ExaTronKernels.dspcg(n, x, xl, xu, tron_A, g, delta, rtol, s, 5, cg_itermax,
            tron_B, tron_L, indfree, gfree, w, wa, iwa)

        @test norm(x .- h_x) <= 1e-10
    end
end

@testset "dgpnorm" begin
    function dgpnorm_test(dx, dxl, dxu, dg, d_out)
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        x = CuDynamicSharedArray(Float64, n)
        xl = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))
        xu = CuDynamicSharedArray(Float64, n, (2 * n) * sizeof(Float64))
        g = CuDynamicSharedArray(Float64, n, (3 * n) * sizeof(Float64))

        x[tx] = dx[tx]
        xl[tx] = dxl[tx]
        xu[tx] = dxu[tx]
        g[tx] = dg[tx]
        CUDA.sync_threads()

        v = dgpnorm(n, x, xl, xu, g)
        if bx == 1
            d_out[tx] = v
        end
        CUDA.sync_threads()

        return
    end

    for i = 1:itermax
        x = rand(n)
        xl = x .- abs.(rand(n))
        xu = x .+ abs.(rand(n))
        g = 2.0 * rand(n) .- 1.0

        dx = CuArray{Float64}(undef, n)
        dxl = CuArray{Float64}(undef, n)
        dxu = CuArray{Float64}(undef, n)
        dg = CuArray{Float64}(undef, n)
        d_out = CuArray{Float64}(undef, n)

        copyto!(dx, x)
        copyto!(dxl, xl)
        copyto!(dxu, xu)
        copyto!(dg, g)

        b = @benchmark(CUDA.@sync @cuda threads = $n blocks = $nblk shmem = (4 * $n * sizeof(Float64)) $dgpnorm_test($dx, $dxl, $dxu, $dg, $d_out))
        display(b)
        save_benchmark(b, "dgpnorm.json")
        h_v = zeros(n)
        copyto!(h_v, d_out)

        v = ExaTronKernels.dgpnorm(n, x, xl, xu, g)
        @test norm(h_v .- v) <= 1e-10
    end
end

@testset "dtron" begin
    function dtron_test(f::Float64, frtol::Float64, fatol::Float64, fmin::Float64,
        cgtol::Float64, cg_itermax::Int, delta::Float64, task::Int,
        disave::CuDeviceArray{Int}, ddsave::CuDeviceArray{Float64},
        dx::CuDeviceArray{Float64}, dxl::CuDeviceArray{Float64},
        dxu::CuDeviceArray{Float64}, dA::CuDeviceArray{Float64},
        dg::CuDeviceArray{Float64}, d_out::CuDeviceArray{Float64})
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        x = CuDynamicSharedArray(Float64, n)
        xl = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))
        xu = CuDynamicSharedArray(Float64, n, (2 * n) * sizeof(Float64))
        g = CuDynamicSharedArray(Float64, n, (3 * n) * sizeof(Float64))
        xc = CuDynamicSharedArray(Float64, n, (4 * n) * sizeof(Float64))
        s = CuDynamicSharedArray(Float64, n, (5 * n) * sizeof(Float64))
        wa = CuDynamicSharedArray(Float64, n, (6 * n) * sizeof(Float64))
        wa1 = CuDynamicSharedArray(Float64, n, (7 * n) * sizeof(Float64))
        wa2 = CuDynamicSharedArray(Float64, n, (8 * n) * sizeof(Float64))
        wa3 = CuDynamicSharedArray(Float64, n, (9 * n) * sizeof(Float64))
        wa4 = CuDynamicSharedArray(Float64, n, (10 * n) * sizeof(Float64))
        wa5 = CuDynamicSharedArray(Float64, n, (11 * n) * sizeof(Float64))
        gfree = CuDynamicSharedArray(Float64, n, (12 * n) * sizeof(Float64))
        indfree = CuDynamicSharedArray(Int, n, (13 * n) * sizeof(Float64))
        iwa = CuDynamicSharedArray(Int, 2 * n, n * sizeof(Int) + (13 * n) * sizeof(Float64))

        A = CuDynamicSharedArray(Float64, (n, n), (13 * n) * sizeof(Float64) + (3 * n) * sizeof(Int))
        B = CuDynamicSharedArray(Float64, (n, n), (13 * n + n^2) * sizeof(Float64) + (3 * n) * sizeof(Int))
        L = CuDynamicSharedArray(Float64, (n, n), (13 * n + 2 * n^2) * sizeof(Float64) + (3 * n) * sizeof(Int))

        @inbounds for j = 1:n
            A[j, tx] = dA[j, tx]
        end
        x[tx] = dx[tx]
        xl[tx] = dxl[tx]
        xu[tx] = dxu[tx]
        g[tx] = dg[tx]
        CUDA.sync_threads()

        dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
            cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
            disave, ddsave, wa, iwa, wa1, wa2, wa3, wa4, wa5)

        if bx == 1
            d_out[tx] = x[tx]
        end
        CUDA.sync_threads()

        return
    end

    for i = 1:itermax
        L = tril(rand(n, n))
        A = L * transpose(L)
        A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
        x = rand(n)
        xl = x .- abs.(rand(n))
        xu = x .+ abs.(rand(n))
        c = rand(n)
        g = A * x .+ c
        xc = zeros(n)
        s = zeros(n)
        wa = zeros(7 * n)
        gfree = zeros(n)
        indfree = zeros(Int, n)
        iwa = zeros(Int, 3 * n)
        isave = zeros(Int, 3)
        dsave = zeros(3)
        task = 0
        fatol = 0.0
        frtol = 1e-12
        fmin = -1e32
        cgtol = 0.1
        cg_itermax = n
        delta = 2.0 * norm(g)
        f = 0.5 * transpose(x) * A * x .+ transpose(x) * c

        tron_A = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
        tron_A.vals .= A
        tron_B = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
        tron_L = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)

        dx = CuArray{Float64}(undef, n)
        dxl = CuArray{Float64}(undef, n)
        dxu = CuArray{Float64}(undef, n)
        dA = CuArray{Float64,2}(undef, (n, n))
        dg = CuArray{Float64}(undef, n)
        disave = CuArray{Int}(undef, n)
        ddsave = CuArray{Float64}(undef, n)
        d_out = CuArray{Float64}(undef, n)

        copyto!(dx, x)
        copyto!(dxl, xl)
        copyto!(dxu, xu)
        copyto!(dA, tron_A.vals)
        copyto!(dg, g)

        b = @benchmark (@cuda threads = $n blocks = $nblk shmem = ((3 * n) * sizeof(Int) + (13 * $n + 3 * ($n^2)) * sizeof(Float64)) $dtron_test($f, $frtol, $fatol, $fmin, $cgtol, $cg_itermax, $delta, $task, $disave, $ddsave, $dx, $dxl, $dxu, $dA, $dg, $d_out))
        display(b)
        save_benchmark(b, "dtron.json")
        h_x = zeros(n)
        copyto!(h_x, d_out)

        task_str = Vector{UInt8}(undef, 60)
        for (i, s) in enumerate("START")
            task_str[i] = UInt8(s)
        end

        ExaTronKernels.dtron(n, x, xl, xu, f, g, tron_A, frtol, fatol, fmin, cgtol,
            cg_itermax, delta, task_str, tron_B, tron_L, xc, s, indfree,
            isave, dsave, wa, iwa)
        @test norm(x .- h_x) <= 1e-10
    end
end

@testset "driver_kernel" begin
    function eval_f(n, x, dA, dc)
        f = 0
        @inbounds for i = 1:n
            @inbounds for j = 1:n
                f += x[i] * dA[i, j] * x[j]
            end
        end
        f = 0.5 * f
        @inbounds for i = 1:n
            f += x[i] * dc[i]
        end
        CUDA.sync_threads()
        return f
    end

    function eval_g(n, x, g, dA, dc)
        @inbounds for i = 1:n
            gval = 0
            @inbounds for j = 1:n
                gval += dA[i, j] * x[j]
            end
            g[i] = gval + dc[i]
        end
        CUDA.sync_threads()
        return
    end

    function eval_h(n, scale, x, A, dA)
        tx = CUDA.threadIdx().x

        @inbounds for j = 1:n
            A[j, tx] = dA[j, tx]
        end
        CUDA.sync_threads()
        return
    end

    function driver_kernel(n::Int, max_feval::Int, max_minor::Int,
        x::CuDeviceArray{Float64}, xl::CuDeviceArray{Float64},
        xu::CuDeviceArray{Float64}, dA::CuDeviceArray{Float64},
        dc::CuDeviceArray{Float64})
        # We start with a shared memory allocation.
        # The first 3*n*sizeof(Float64) bytes are used for x, xl, and xu.

        g = CuDynamicSharedArray(Float64, n, (3 * n) * sizeof(Float64))
        xc = CuDynamicSharedArray(Float64, n, (4 * n) * sizeof(Float64))
        s = CuDynamicSharedArray(Float64, n, (5 * n) * sizeof(Float64))
        wa = CuDynamicSharedArray(Float64, n, (6 * n) * sizeof(Float64))
        wa1 = CuDynamicSharedArray(Float64, n, (7 * n) * sizeof(Float64))
        wa2 = CuDynamicSharedArray(Float64, n, (8 * n) * sizeof(Float64))
        wa3 = CuDynamicSharedArray(Float64, n, (9 * n) * sizeof(Float64))
        wa4 = CuDynamicSharedArray(Float64, n, (10 * n) * sizeof(Float64))
        wa5 = CuDynamicSharedArray(Float64, n, (11 * n) * sizeof(Float64))
        gfree = CuDynamicSharedArray(Float64, n, (12 * n) * sizeof(Float64))
        dsave = CuDynamicSharedArray(Float64, n, (13 * n) * sizeof(Float64))
        indfree = CuDynamicSharedArray(Int, n, (14 * n) * sizeof(Float64))
        iwa = CuDynamicSharedArray(Int, 2 * n, n * sizeof(Int) + (14 * n) * sizeof(Float64))
        isave = CuDynamicSharedArray(Int, n, (3 * n) * sizeof(Int) + (14 * n) * sizeof(Float64))

        A = CuDynamicSharedArray(Float64, (n, n), (14 * n) * sizeof(Float64) + (4 * n) * sizeof(Int))
        B = CuDynamicSharedArray(Float64, (n, n), (14 * n + n^2) * sizeof(Float64) + (4 * n) * sizeof(Int))
        L = CuDynamicSharedArray(Float64, (n, n), (14 * n + 2 * n^2) * sizeof(Float64) + (4 * n) * sizeof(Int))

        task = 0
        status = 0

        delta = 0.0
        fatol = 0.0
        frtol = 1e-12
        fmin = -1e32
        gtol = 1e-6
        cgtol = 0.1
        cg_itermax = n

        f = 0.0
        nfev = 0
        ngev = 0
        nhev = 0
        minor_iter = 0
        search = true

        while search

            # [0|1]: Evaluate function.

            if task == 0 || task == 1
                f = eval_f(n, x, dA, dc)
                nfev += 1
                if nfev >= max_feval
                    search = false
                end
            end

            # [2] G or H: Evaluate gradient and Hessian.

            if task == 0 || task == 2
                eval_g(n, x, g, dA, dc)
                eval_h(n, 1.0, x, A, dA)
                ngev += 1
                nhev += 1
                minor_iter += 1
            end

            # Initialize the trust region bound.

            if task == 0
                gnorm0 = dnrm2(n, g, 1)
                delta = gnorm0
            end

            # Call Tron.

            if search
                delta, task = dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                    cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                    isave, dsave, wa, iwa, wa1, wa2, wa3, wa4, wa5)
            end

            # [3] NEWX: a new point was computed.

            if task == 3
                gnorm_inf = dgpnorm(n, x, xl, xu, g)
                if gnorm_inf <= gtol
                    task = 4
                end

                if minor_iter >= max_minor
                    status = 1
                    search = false
                end
            end

            # [4] CONV: convergence was achieved.

            if task == 4
                search = false
            end
        end

        return status, minor_iter
    end

    function driver_kernel_test(max_feval, max_minor,
        dx, dxl, dxu, dA, dc, d_out)
        tx = CUDA.threadIdx().x
        bx = CUDA.blockIdx().x

        x = CuDynamicSharedArray(Float64, n)
        xl = CuDynamicSharedArray(Float64, n, n * sizeof(Float64))
        xu = CuDynamicSharedArray(Float64, n, (2 * n) * sizeof(Float64))

        x[tx] = dx[tx]
        xl[tx] = dxl[tx]
        xu[tx] = dxu[tx]
        CUDA.sync_threads()

        status, minor_iter = driver_kernel(n, max_feval, max_minor, x, xl, xu, dA, dc)

        if bx == 1
            d_out[tx] = x[tx]
        end
        CUDA.sync_threads()
        return
    end

    max_feval = 500
    max_minor = 100

    tron_A = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
    tron_B = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)
    tron_L = ExaTronKernels.TronDenseMatrix{Array{Float64,2}}(n)

    dx = CuArray{Float64}(undef, n)
    dxl = CuArray{Float64}(undef, n)
    dxu = CuArray{Float64}(undef, n)
    dA = CuArray{Float64,2}(undef, (n, n))
    dc = CuArray{Float64}(undef, n)
    d_out = CuArray{Float64}(undef, n)

    for i = 1:itermax
        L = tril(rand(n, n))
        A = L * transpose(L)
        A .= tril(A) .+ (transpose(tril(A)) .- Diagonal(A))
        x = rand(n)
        xl = x .- abs.(rand(n))
        xu = x .+ abs.(rand(n))
        c = rand(n)

        tron_A.vals .= A

        copyto!(dx, x)
        copyto!(dxl, xl)
        copyto!(dxu, xu)
        copyto!(dA, tron_A.vals)
        copyto!(dc, c)

        function eval_f_cb(x)
            f = 0.5 * (transpose(x) * A * x) + transpose(c) * x
            return f
        end

        function eval_g_cb(x, g)
            g .= A * x .+ c
        end

        function eval_h_cb(x, mode, rows, cols, scale, lambda, values)
            nz = 1
            if mode == :Structure
                @inbounds for j = 1:n
                    @inbounds for i = j:n
                        rows[nz] = i
                        cols[nz] = j
                        nz += 1
                    end
                end
            else
                @inbounds for j = 1:n
                    @inbounds for i = j:n
                        values[nz] = A[i, j]
                        nz += 1
                    end
                end
            end
        end

        nele_hess = div(n * (n + 1), 2)
        tron = ExaTronKernels.createProblem(n, xl, xu, nele_hess, eval_f_cb, eval_g_cb, eval_h_cb; :matrix_type => :Dense, :max_minor => max_minor)
        copyto!(tron.x, x)
        status = ExaTronKernels.solveProblem(tron)

        b = @benchmark(CUDA.@sync @cuda threads = $n blocks = $nblk shmem = ((4 * $n) * sizeof(Int) + (14 * $n + 3 * ($n^2)) * sizeof(Float64)) $driver_kernel_test($max_feval, $max_minor, $dx, $dxl, $dxu, $dA, $dc, $d_out))
        display(b)
        save_benchmark(b, "driver_kernel.json")
        h_x = zeros(n)
        copyto!(h_x, d_out)

        @test norm(h_x .- tron.x) <= 1e-10
    end
end
