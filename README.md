# tinyvec

A fixed-capacity vector type for Rust with just a single byte of overhead.
Supports a useful subset of `Vec`'s API.

## Usage

```rust
use tinyvec::TinyVec;

// Create a vector with up to 4 elements of type u8
let vec = TinyVec::<u8, 4>::new();
// push up to 4 elements
vec.push(1);
vec.push(2);
vec.push(3);
vec.push(4);
// this will panic:
vec.push(5);
```

## Limitations

Stores at most 255 elements. Attempting to `push` when the vector is at capacity will result in a `panic`.
