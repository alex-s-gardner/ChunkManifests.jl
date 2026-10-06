using Aqua
using ExplicitImports
import HDF5
import HTTP
import TiffImages
import JSON

# Last in the suite, so every extension is loaded by the time this runs and the
# ambiguity and piracy checks cover their methods as well as the base module's.
@testset "Aqua" begin
    Aqua.test_all(
        ChunkManifests;
        # Aqua walks the test manifest and throws on SymDict, an AWSS3
        # dependency carrying only a REQUIRE file. Recorded in UPSTREAM.md.
        persistent_tasks = false,
        # Aqua leaves this off by default. On here, an undocumented exported
        # name fails the suite. It has teeth only on Julia 1.11 and later,
        # where Docs.undocumented_names exists.
        undocumented_names = true,
    )
end

@testset "ExplicitImports" begin
    test_explicit_imports(
        ChunkManifests;
        # Only accurate from Julia 1.11, where "public" means Base.ispublic
        # rather than falling back to isexported.
        all_explicit_imports_are_public = VERSION >= v"1.11",
        # Three reasons this fails, none fixable without reimplementing
        # another package's internals or over-exposing our own. Zarr exports
        # no API for writing a custom store or filter — building
        # ChunkManifest <: Zarr.AbstractStore and the TIFFPredictor codec
        # means reaching into AbstractStore, storefromstring, read_items!,
        # Filter, zdecode/zencode and the compressor types directly. HDF5's
        # public API has no chunk-layout introspection, so scanning chunk
        # offsets without reading data means HDF5.API ccalls and
        # get_chunk_info_all. Parquet2 exports nothing but its own module
        # name, so every call into it is qualified by design. The
        # Parquet2 extension also calls back into ChunkManifests' own
        # unexported helpers (zarray_json, save, ...) to build the metadata
        # it writes — an extension and its parent are one package split
        # across a module boundary Julia itself imposes, so this is no
        # different from one file in src/ calling a `_`-prefixed function
        # in another.
        all_qualified_accesses_are_public = false,
        # HTTP.get is Base.get — HTTP.jl extends it rather than defining its
        # own name — and JSON.lower resolves to StructUtils.lower on every
        # Julia version checked. The check only fails on Julia 1.10, not on
        # release, despite that identical fact; why its verdict differs by
        # version was not fully run down. Either way the call sites are
        # HTTP.jl's and JSON.jl's documented entry points, not a mistake
        # here, so skip (not ignore) exempts only these exact
        # (accessing-module, owner) pairs rather than `get`/`lower`
        # package-wide.
        # HDF5.API.libhdf5 is owned by HDF5_jll but is how HDF5.jl documents
        # naming the library in a ccall, which the range virtual file driver
        # has to do: libhdf5 exposes no public API for registering one, so
        # there is no wrapper in HDF5.jl to call instead. TiffImages.format"TIFF"
        # is FileIO's, and is the stream wrapper TiffImages itself reads a
        # TiffFile from, so the GeoTIFF driver names it the same way.
        all_qualified_accesses_via_owners = (;
            skip = (
                Base => Core, HTTP => Base, JSON => parentmodule(JSON.lower),
                HDF5.API => HDF5.API.HDF5_jll,
                TiffImages => TiffImages.FileIO,
            ),
        ),
    )
end
