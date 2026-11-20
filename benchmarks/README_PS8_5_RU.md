# PS8.5 reference benchmark

C++ reference implementation mirrors the PS8.5 sampled profile selection: test depths 32/48/64 on the first <=64 KiB, choose the smallest PYLZ sample (ties prefer the lower depth), then encode the full input once. This is a reference harness, not a Rust build.
