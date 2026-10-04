using Aqua

# Last in the suite, so every extension is loaded by the time this runs and the
# ambiguity and piracy checks cover their methods as well as the base module's.
@testset "Aqua" begin
    Aqua.test_all(
        ChunkManifests;
        # Aqua walks the test manifest and throws on SymDict, an AWSS3
        # dependency carrying only a REQUIRE file. Recorded in UPSTREAM.md.
        persistent_tasks=false,
        # Aqua leaves this off by default. On here, an undocumented exported
        # name fails the suite. It has teeth only on Julia 1.11 and later,
        # where Docs.undocumented_names exists.
        undocumented_names=true,
    )
end
