# NODEData

Small helper package that provides a struct for sequence learning with Neural ODEs. It behaves roughly similar to Flux' Dataloader but the individual samples overlap, so that it is suitable for learning sequences.

## Usage

Prepare a DE solution

```julia
f(u,p,t) = 1.01*u
u0 = 1/2
tspan = (0.0,10.0)
prob = ODEProblem(f,u0,tspan)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
```
and either interpolate the result
```julia
data = NODEDataloader(sol, 20, dt=0.2)
```

or use their original timesteps
```julia
data = NODEDataloader(sol, 20)
```

In these examples each batch is `N_length=20` elements long, i.e `data[i]`, is a tuple with `(t, data(t))` each with 20 elements. `data[1]` are the first `N_length` elements, `data[2]` are the `2:N_length+1` elements and so on.

### Larger than RAM data

The pacakge also provides a wrapper around `NODEDataloader` for larger than RAM datasets. The data is split into temporary files on the harddrive and can be easiliy loaded. See `LargeNODEDataloader`