# TIFF's LZW compression (TIFF 6.0 section 13), decoded for typed opens.
#
# The variant every TIFF writer has used since TIFF 5.0: codes are packed most significant bit first,
# start at 9 bits, and widen one code *early* — at table sizes 511, 1023 and 2047, not 512, 1024 and
# 2048 — as libtiff does. Code 256 clears the table and 257 ends the strip. The pre-5.0 variant, packed
# least significant bit first, begins with the bytes `0x00 0x01`; libtiff reads it as "old-style" LZW,
# and it is refused here rather than decoded wrongly.

const _LZW_CLEAR = 256
const _LZW_EOI = 257
const _LZW_FIRST = 258
const _LZW_MAXCODE = 4096

"""
    lzw_decode!(out, src) -> Int

Decodes the TIFF LZW stream `src` into the front of `out`, returning the number of bytes written. A
stream decoding to more bytes than `out` holds is an error, so a corrupt stream cannot write past the
chunk; one decoding to fewer leaves the rest of `out` as it was, which is what the last strip of a
striped image needs.
"""
function lzw_decode!(out::AbstractVector{UInt8}, src::AbstractVector{UInt8})
    length(src) >= 2 && src[begin] == 0x00 && (src[begin + 1] & 0x01) == 0x01 &&
        throw(ArgumentError("an old-style (pre-TIFF 5.0, LSB-first) LZW stream is not decoded"))
    # Each table entry is its prefix code and last byte; entry length gives where it lands in `out`.
    prefix = zeros(Int16, _LZW_MAXCODE)
    suffix = zeros(UInt8, _LZW_MAXCODE)
    lengths = zeros(Int32, _LZW_MAXCODE)
    for c in 0:255
        suffix[c + 1] = UInt8(c); lengths[c + 1] = 1; prefix[c + 1] = -1
    end
    next = _LZW_FIRST
    width = 9
    acc = UInt32(0); nacc = 0
    pos = firstindex(src)
    o = firstindex(out) - 1
    old = -1
    while true
        while nacc < width
            pos > lastindex(src) && return o - (firstindex(out) - 1)    # no EOI: the stream ends with the data
            acc = (acc << 8) | src[pos]; pos += 1; nacc += 8
        end
        code = Int((acc >> (nacc - width)) & ((UInt32(1) << width) - 1))
        nacc -= width
        if code == _LZW_EOI
            break
        elseif code == _LZW_CLEAR
            next = _LZW_FIRST; width = 9; old = -1
            continue
        end
        if old == -1
            code < 256 || throw(ArgumentError("an LZW stream's first code after a clear is $code, not a literal"))
            o + 1 > lastindex(out) && throw(ArgumentError("an LZW stream decodes to more than its chunk's $(length(out)) bytes"))
            o += 1; out[o] = UInt8(code)
            old = code
            continue
        end
        if code < next
            # Known code: write its string, then add old + first byte of this one.
            n = lengths[code + 1]
            o + n > lastindex(out) && throw(ArgumentError("an LZW stream decodes to more than its chunk's $(length(out)) bytes"))
            c = code
            for k in n:-1:1
                out[o + k] = suffix[c + 1]; c = prefix[c + 1]
            end
            first = out[o + 1]
            o += n
        elseif code == next
            # The KwKwK case: old's string followed by its own first byte.
            n = lengths[old + 1] + 1
            o + n > lastindex(out) && throw(ArgumentError("an LZW stream decodes to more than its chunk's $(length(out)) bytes"))
            c = old
            for k in (n - 1):-1:1
                out[o + k] = suffix[c + 1]; c = prefix[c + 1]
            end
            first = out[o + 1]
            out[o + n] = first
            o += n
        else
            throw(ArgumentError("an LZW code $code is beyond the table's $next entries"))
        end
        if next < _LZW_MAXCODE
            prefix[next + 1] = old; suffix[next + 1] = first; lengths[next + 1] = lengths[old + 1] + 1
            next += 1
            # Early change: widen when the *next* code would not fit, one entry before the table fills.
            next + 1 >= (1 << width) && width < 12 && (width += 1)
        end
        old = code
    end
    return o - (firstindex(out) - 1)
end
