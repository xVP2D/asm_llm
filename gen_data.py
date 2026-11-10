#!/usr/bin/env python3
import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

WORDS = ["alpha", "bravo", "nasm", "shell", "kernel", "stack", "heap", "byte", "word", "loop", "jump",
         "match", "done", "access", "ready", "hello", "world", "win", "yes", "ok", "bingo", "pong",
         "ping", "magic", "x0ne", "linux", "code", "data", "flag", "root", "zero", "one", "light",
         "blue", "green", "red", "cat", "dog", "fox", "owl", "sun", "moon", "star", "bit", "cpu",
         "ram", "disk", "file", "path", "node", "tree", "leaf", "gate", "door", "key", "lock"]
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}

ATOI = """
atoi:
    xor eax, eax
    xor ecx, ecx
    cmp byte [rsi], '-'
    jne .first
    inc ecx
    inc rsi
.first:
    movzx edx, byte [rsi]
    sub edx, '0'
    cmp edx, 9
    ja .bad
.loop:
    movzx edx, byte [rsi]
    test edx, edx
    jz .done
    cmp edx, 10
    je .done
    sub edx, '0'
    cmp edx, 9
    ja .bad
    imul rax, rax, 10
    jo .bad
    add rax, rdx
    jo .bad
    inc rsi
    jmp .loop
.done:
    test ecx, ecx
    jz .ok
    neg rax
.ok:
    clc
    ret
.bad:
    stc
    ret"""

PRINT_NUM = """
print_num:
    mov r8, rax
    mov rsi, numbuf + 31
    mov byte [rsi], 10
    mov ecx, 1
    test rax, rax
    jns .conv
    neg rax
.conv:
    xor edx, edx
    mov r9, 10
    div r9
    add dl, '0'
    dec rsi
    mov [rsi], dl
    inc ecx
    test rax, rax
    jnz .conv
    test r8, r8
    jns .write
    dec rsi
    mov byte [rsi], '-'
    inc ecx
.write:
    mov eax, 1
    mov edi, 1
    mov edx, ecx
    syscall
    ret"""

READ_NUM = """    xor eax, eax
    xor edi, edi
    mov rsi, buf
    mov edx, 64
    syscall
    test rax, rax
    jle invalid
    mov byte [buf + rax], 0
    mov rsi, buf
    call atoi
    jc invalid"""

EXIT = """exit:
    mov eax, 60
    syscall"""


def word(rng, avoid=()):
    while True:
        w = rng.choice(WORDS) if rng.random() < 0.7 else "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 6)))
        if w not in avoid:
            return w


def phrase(rng):
    return " ".join(word(rng) for _ in range(rng.randint(1, 3)))


def codes(rng, n, fixed=None):
    if fixed and rng.random() < 0.5:
        return fixed
    pool = list(range(0, 10))
    rng.shuffle(pool)
    return pool[:n]


def gen_exit(rng):
    n = rng.choice([rng.randint(0, 9), rng.randint(0, 255)])
    st = rng.choice([
        "Create a program that exits with code {n}.",
        "Write a program that returns the exit status {n}.",
        "Make a program whose exit code is {n}.",
        "Exit immediately with status {n}.",
        "The program prints nothing and terminates with exit code {n}.",
        "Write a program that does nothing except exit with code {n}.",
    ]).format(n=n)
    src = f"""section .text
    global _start

_start:
    mov eax, 60
    mov edi, {n}
    syscall
"""
    return st, src, [("./prog", n, "")]


def gen_print(rng):
    m = phrase(rng)
    c = rng.choice([0, 0, 0, rng.randint(0, 9)])
    st = rng.choice([
        "Print '{m}' followed by a newline, then exit {c}.",
        "Create a program that prints '{m}' and exits with code {c}.",
        "Display '{m}' on standard output with a trailing newline and exit with status {c}.",
        "Write '{m}' to stdout, then a newline, and exit {c}.",
        "Output the text '{m}' and a newline, then terminate with exit code {c}.",
    ]).format(m=m, c=c)
    src = f"""section .data
    msg db '{m}', 10
    len equ $ - msg

section .text
    global _start

_start:
    mov eax, 1
    mov edi, 1
    mov rsi, msg
    mov edx, len
    syscall
    mov eax, 60
    mov edi, {c}
    syscall
"""
    return st, src, [("./prog", c, m + "\n")]


def gen_stdin_eq(rng):
    w = word(rng)
    m = phrase(rng)
    ok, ko = codes(rng, 2, [0, 1])
    st = rng.choice([
        "Read from stdin. If input is '{w}', print '{m}' and exit {ok}. Otherwise exit {ko}.",
        "Read a line from standard input. When it equals '{w}', write '{m}' and exit {ok}; any other input exits {ko} without output.",
        "If the text read on stdin is exactly '{w}', display '{m}' and exit {ok}. Anything else, including empty input, exits {ko} and prints nothing.",
        "Read from stdin. If the input is exactly '{w}', print '{m}' and exit {ok}. Anything else (other text, empty input) exits {ko} and prints nothing.",
    ]).format(w=w, m=m, ok=ok, ko=ko)
    src = f"""section .data
    expected db '{w}'
    elen equ $ - expected
    msg db '{m}', 10
    mlen equ $ - msg

section .bss
    buf resb 72

section .text
    global _start

_start:
    xor eax, eax
    xor edi, edi
    mov rsi, buf
    mov edx, 64
    syscall
    test rax, rax
    jle fail
    cmp byte [buf + rax - 1], 10
    jne check
    dec rax
check:
    cmp rax, elen
    jne fail
    mov rsi, buf
    mov rdi, expected
    mov ecx, elen
    repe cmpsb
    jne fail
    mov eax, 1
    mov edi, 1
    mov rsi, msg
    mov edx, mlen
    syscall
    mov edi, {ok}
    jmp exit
fail:
    mov edi, {ko}
{EXIT}
"""
    tests = [(f"echo {w} | ./prog", ok, m + "\n"), (f"echo {w}x | ./prog", ko, ""),
             (f"echo {w[:-1]} | ./prog", ko, ""), ("echo '' | ./prog", ko, ""), ("printf '' | ./prog", ko, "")]
    return st, src, tests


def gen_arg_eq(rng):
    w = word(rng, avoid=("abc",))
    m = phrase(rng)
    ok, ko, e = codes(rng, 3, [0, 1, 2])
    st = rng.choice([
        "If the program's only argument is exactly '{w}', print '{m}' and exit {ok}. If it is different, exit {ko} without printing. Exit {e} unless there is exactly one argument.",
        "Check if the first argument equals '{w}'. If yes, print '{m}' and exit {ok}; otherwise exit {ko} and print nothing. Exactly one argument: no argument or more than one exits {e}.",
        "Compare the single command-line argument with '{w}': on a match display '{m}' and exit {ok}, on a mismatch exit {ko} silently. Any other number of arguments exits {e}.",
    ]).format(w=w, m=m, ok=ok, ko=ko, e=e)
    src = f"""section .data
    expected db '{w}', 0
    elen equ $ - expected
    msg db '{m}', 10
    mlen equ $ - msg

section .text
    global _start

_start:
    cmp qword [rsp], 2
    jne usage
    mov rsi, [rsp + 16]
    mov rdi, expected
    mov ecx, elen
    repe cmpsb
    jne fail
    mov eax, 1
    mov edi, 1
    mov rsi, msg
    mov edx, mlen
    syscall
    mov edi, {ok}
    jmp exit
fail:
    mov edi, {ko}
    jmp exit
usage:
    mov edi, {e}
{EXIT}
"""
    tests = [(f"./prog {w}", ok, m + "\n"), ("./prog abc", ko, ""), (f"./prog {w}z", ko, ""),
             ("./prog", e, ""), (f"./prog {w} {w}", e, "")]
    return st, src, tests


def num_program(body, extra_bss="", tail=""):
    return f"""section .bss
    buf resb 72{extra_bss}

section .text
    global _start

_start:
{READ_NUM}
{body}
{EXIT}
{ATOI}{tail}
"""


def stdin_cmd(v):
    return f"echo {v} | ./prog"


def gen_divisible(rng):
    k = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 20, 25, 30, 50, 100])
    ok, ko, inv = codes(rng, 3, [0, 1, 2])
    st = rng.choice([
        "Read a number from stdin. Exit {ok} if it is divisible by {k}, exit {ko} if not, exit {inv} if it is not a number.",
        "Read an integer on standard input and exit {ok} when it is a multiple of {k}, {ko} otherwise. Invalid input exits {inv}.",
        "Check whether the signed number read from stdin is divisible by {k}: exit {ok} if yes, {ko} if no, {inv} if the input is not a valid number.",
    ]).format(k=k, ok=ok, ko=ko, inv=inv)
    body = f"""    cqo
    mov rcx, {k}
    idiv rcx
    test rdx, rdx
    jnz no
    mov edi, {ok}
    jmp exit
no:
    mov edi, {ko}
    jmp exit
invalid:
    mov edi, {inv}"""
    a = k * rng.randint(2, 500)
    tests = [(stdin_cmd(a), ok, None), (stdin_cmd(a + 1), ko, None), (stdin_cmd(-a), ok, None),
             (stdin_cmd(0), ok, None), ("echo abc | ./prog", inv, ""), ("echo 12x | ./prog", inv, ""),
             ("printf '' | ./prog", inv, "")]
    return st, num_program(body), tests


def gen_parity(rng):
    ev, od, inv = codes(rng, 3, [0, 1, 2])
    st = rng.choice([
        "Read an integer from standard input. Exit with {od} when it is odd and {ev} when it is even. Invalid input exits {inv}.",
        "Read a number from stdin. Exit {ev} if even, exit {od} if odd, exit {inv} if it is not a number. The number may be negative.",
        "Tell whether the signed integer on stdin is even (exit {ev}) or odd (exit {od}). Exit {inv} when the input is not a number.",
    ]).format(ev=ev, od=od, inv=inv)
    body = f"""    test al, 1
    jnz odd
    mov edi, {ev}
    jmp exit
odd:
    mov edi, {od}
    jmp exit
invalid:
    mov edi, {inv}"""
    a = rng.randint(1, 5000) * 2
    tests = [(stdin_cmd(a), ev, None), (stdin_cmd(a + 1), od, None), (stdin_cmd(0), ev, None),
             (stdin_cmd(-a), ev, None), (stdin_cmd(-a - 1), od, None), ("echo abc | ./prog", inv, ""),
             ("printf '' | ./prog", inv, "")]
    return st, num_program(body), tests


PROPS = [
    ("greater than {x}", "jg", lambda v, x: v > x),
    ("less than {x}", "jl", lambda v, x: v < x),
    ("at least {x}", "jge", lambda v, x: v >= x),
    ("at most {x}", "jle", lambda v, x: v <= x),
    ("equal to {x}", "je", lambda v, x: v == x),
    ("strictly positive", "jg", lambda v, x: v > 0),
    ("negative", "jl", lambda v, x: v < 0),
]


def gen_property(rng):
    name, jcc, f = rng.choice(PROPS)
    x = 0 if "{x}" not in name else rng.choice([rng.randint(-100, 100), rng.randint(0, 1000)])
    desc = name.format(x=x)
    ok, ko, inv = codes(rng, 3, [0, 1, 2])
    st = rng.choice([
        "Read a signed number on stdin. If it is {d} exit {ok}, if it is not exit {ko}, and exit {inv} on invalid input.",
        "Read an integer from standard input. Exit {ok} when it is {d}, {ko} otherwise, {inv} if the input is not a number.",
        "Read a number from stdin and exit {ok} if it is {d}. Otherwise exit {ko}. Exit {inv} if the input is not a valid integer.",
    ]).format(d=desc, ok=ok, ko=ko, inv=inv)
    body = f"""    cmp rax, {x}
    {jcc} yes
    mov edi, {ko}
    jmp exit
yes:
    mov edi, {ok}
    jmp exit
invalid:
    mov edi, {inv}"""
    vals = sorted({x - 1, x, x + 1, x + rng.randint(2, 500), x - rng.randint(2, 500), 0})
    tests = [(f"printf -- '{v}\\n' | ./prog", ok if f(v, x) else ko, None) for v in vals]
    tests += [("printf 'abc\\n' | ./prog", inv, ""), ("printf '' | ./prog", inv, "")]
    return st, num_program(body), tests


def strlen_block():
    return """    xor edx, edx
len:
    cmp byte [rsi + rdx], 0
    je write
    inc rdx
    jmp len
write:"""


def gen_print_arg(rng):
    total = rng.randint(1, 6)
    n = rng.randint(1, total)
    e = rng.choice([1, 2, 3, 2, 3])
    if total == 1:
        st = rng.choice([
            "Print the string passed as argument. Exactly one argument: anything else exits {e} and prints nothing.",
            "Write the only command-line argument to stdout followed by a newline. Exit {e} unless there is exactly one argument.",
            "Echo the single argument given to the program, with a trailing newline. Wrong argument count: exit {e}.",
        ]).format(e=e)
    else:
        st = rng.choice([
            "Write the {o} command-line argument to stdout with a trailing newline. Exactly {t} arguments are expected, exit {e} otherwise.",
            "Display argument number {n} followed by a newline. The program needs exactly {t} arguments; otherwise it exits {e} without output.",
            "Given exactly {t} arguments, print the {o} one and a newline. Any other argument count exits {e}.",
        ]).format(o=ORD[n], n=n, t=total, e=e)
    src = f"""section .data
    nl db 10

section .text
    global _start

_start:
    cmp qword [rsp], {total + 1}
    jne usage
    mov rsi, [rsp + {8 * (n + 1)}]
{strlen_block()}
    mov eax, 1
    mov edi, 1
    syscall
    mov eax, 1
    mov edi, 1
    mov rsi, nl
    mov edx, 1
    syscall
    xor edi, edi
    jmp exit
usage:
    mov edi, {e}
{EXIT}
"""
    args = [word(rng) for _ in range(total)]
    tests = [("./prog " + " ".join(args), 0, args[n - 1] + "\n"), ("./prog " + " ".join(args + ["extra"]), e, "")]
    if total > 1:
        tests.append(("./prog " + " ".join(args[:-1]), e, ""))
    else:
        tests.append(("./prog", e, ""))
    return st, src, tests


OPS = {
    "min": (["Print the smaller of two numbers given on the command line, print the result followed by a newline.",
             "Display the minimum of the two integers passed as arguments, followed by a newline."],
            "    cmp rax, rbx\n    cmovg rax, rbx", min),
    "max": (["Print the larger of two numbers given on the command line, followed by a newline.",
             "Display the maximum of the two integers passed as arguments, followed by a newline."],
            "    cmp rax, rbx\n    cmovl rax, rbx", max),
    "add": (["Add the two numbers passed as arguments and print the result.",
             "Print the sum of the two integers given on the command line, followed by a newline."],
            "    add rax, rbx\n    jo invalid", lambda a, b: a + b),
    "sub": (["Subtract the second argument from the first and print the difference.",
             "Print the first command-line number minus the second, followed by a newline."],
            "    sub rbx, rax\n    jo invalid\n    mov rax, rbx", lambda a, b: a - b),
    "mul": (["Multiply the two numbers passed as arguments and print the product.",
             "Print the product of the two integers given on the command line, followed by a newline."],
            "    imul rax, rbx\n    jo invalid", lambda a, b: a * b),
}


def arith_tail(rng, c, i):
    return rng.choice([
        " Exit {c} on a wrong argument count, exit {i} on an invalid number or an overflow.",
        " Exactly two arguments (otherwise exit {c}). Numbers may be negative; an invalid number or an overflow exits {i} and prints nothing.",
        " Wrong argument count exits {c}, an invalid number or an overflow exits {i}.",
    ]).format(c=c, i=i)


def two_args_program(op_asm, c, i):
    return f"""section .bss
    numbuf resb 32

section .text
    global _start

_start:
    cmp qword [rsp], 3
    jne usage
    mov rsi, [rsp + 16]
    call atoi
    jc invalid
    mov rbx, rax
    mov rsi, [rsp + 24]
    call atoi
    jc invalid
{op_asm}
    call print_num
    xor edi, edi
    jmp exit
usage:
    mov edi, {c}
    jmp exit
invalid:
    mov edi, {i}
{EXIT}
{ATOI}
{PRINT_NUM}
"""


def gen_arith(rng):
    op = rng.choice(list(OPS))
    sts, op_asm, f = OPS[op]
    c, i = rng.choice([(2, 3), (2, 1), (1, 2), (3, 2), (2, 4)])
    st = rng.choice(sts) + arith_tail(rng, c, i)
    a, b = rng.randint(-500, 500), rng.randint(-500, 500)
    tests = [(f"./prog {a} {b}", 0, f"{f(a, b)}\n"), (f"./prog {b} {a}", 0, f"{f(b, a)}\n"),
             ("./prog 0 0", 0, f"{f(0, 0)}\n"), ("./prog 7", c, ""), ("./prog", c, ""),
             ("./prog abc 3", i, ""), ("./prog 1 2 3", c, "")]
    if op == "add":
        tests.append(("./prog 9223372036854775807 1", i, ""))
    if op == "mul":
        tests.append(("./prog 9223372036854775807 2", i, ""))
    return st, two_args_program(op_asm, c, i), tests


def gen_arg_len(rng):
    c = rng.choice([1, 2, 3, 4, 2])
    st = rng.choice([
        "Print the number of characters in the first argument, then a newline. Exit {c} if the argument count is not one.",
        "Compute the length of the string given as the only argument and display it. Any other argument count exits {c}.",
        "Count the characters of the string passed as first argument and print that number. Wrong argument count: exit {c}.",
        "Print the length of the single command-line argument followed by a newline. Exit {c} unless there is exactly one argument.",
        "Display how many characters the only argument has. Exactly one argument is expected, exit {c} otherwise.",
    ]).format(c=c)
    src = f"""section .bss
    numbuf resb 32

section .text
    global _start

_start:
    cmp qword [rsp], 2
    jne usage
    mov rsi, [rsp + 16]
    xor eax, eax
len:
    cmp byte [rsi + rax], 0
    je done
    inc rax
    jmp len
done:
    call print_num
    xor edi, edi
    jmp exit
usage:
    mov edi, {c}
{EXIT}
{PRINT_NUM}
"""
    s = word(rng) + word(rng)
    tests = [(f"./prog {s}", 0, f"{len(s)}\n"), ("./prog ''", 0, "0\n"), ("./prog", c, ""), ("./prog a b", c, "")]
    return st, src, tests


THREE = {
    "sum": (["Take three command-line integers, possibly negative, and display their total.",
             "Print the sum of the three numbers passed as arguments.",
             "Add the three integers given on the command line and print the result followed by a newline."],
            "    add rbx, rax\n    jo invalid", sum, True),
    "max": (["Print the maximum of the three numbers passed as arguments.",
             "Display the largest of three command-line integers, possibly negative."],
            "    cmp rbx, rax\n    cmovl rbx, rax", max, False),
    "min": (["Print the minimum of the three numbers passed as arguments.",
             "Display the smallest of three command-line integers, possibly negative."],
            "    cmp rbx, rax\n    cmovg rbx, rax", min, False),
}


def gen_three(rng):
    op = rng.choice(["sum", "sum", "max", "min"])
    sts, op_asm, f, ovf = THREE[op]
    c, i = rng.choice([(2, 1), (2, 3), (3, 1), (1, 2)])
    tail = rng.choice([
        " Exit {c} unless there are exactly three, exit {i} if one is not a number" + (" or the sum overflows." if ovf else "."),
        " Exactly three arguments (otherwise exit {c}). An invalid number" + (" or an overflow" if ovf else "") + " exits {i}.",
    ]).format(c=c, i=i)
    st = rng.choice(sts) + tail
    src = f"""section .bss
    numbuf resb 32

section .text
    global _start

_start:
    cmp qword [rsp], 4
    jne usage
    mov rsi, [rsp + 16]
    call atoi
    jc invalid
    mov rbx, rax
    mov rsi, [rsp + 24]
    call atoi
    jc invalid
{op_asm}
    mov rsi, [rsp + 32]
    call atoi
    jc invalid
{op_asm}
    mov rax, rbx
    call print_num
    xor edi, edi
    jmp exit
usage:
    mov edi, {c}
    jmp exit
invalid:
    mov edi, {i}
{EXIT}
{ATOI}
{PRINT_NUM}
"""
    v = [rng.randint(-300, 300) for _ in range(3)]
    fn = (lambda xs: sum(xs)) if op == "sum" else f
    tests = [("./prog " + " ".join(map(str, v)), 0, f"{fn(v)}\n"), ("./prog 0 0 0", 0, "0\n"),
             ("./prog 1 2", c, ""), ("./prog 1 x 3", i, ""), ("./prog", c, "")]
    if ovf:
        tests.append(("./prog 9223372036854775807 1 0", i, ""))
    return st, src, tests


FAMILIES = {"exit": gen_exit, "print": gen_print, "stdin_eq": gen_stdin_eq, "arg_eq": gen_arg_eq,
            "divisible": gen_divisible, "parity": gen_parity, "print_arg": gen_print_arg, "arith": gen_arith,
            "property": gen_property, "arg_len": gen_arg_len, "sum3": gen_three}


def run_tests(src, tests, work):
    (work / "prog.s").write_text(src)
    b = subprocess.run("nasm -f elf64 prog.s -o prog.o && ld prog.o -o prog", shell=True, cwd=work,
                       capture_output=True, text=True)
    if b.returncode:
        return b.stderr.strip()
    for cmd, code, out in tests:
        p = subprocess.run(["bash", "-c", cmd], cwd=work, capture_output=True, text=True, timeout=5)
        if p.returncode != code or (out is not None and p.stdout != out):
            return f"{cmd}: expected {code} {out!r}, got {p.returncode} {p.stdout!r}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-family", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    public = {json.loads(l)["statement"] for l in (HERE / "public.jsonl").read_text().splitlines() if l.strip()}
    rows, seen = [], set(public)
    work = Path(tempfile.mkdtemp(prefix="gen-"))
    for fam, gen in FAMILIES.items():
        n = tries = 0
        while n < a.per_family and tries < 50 * a.per_family:
            tries += 1
            st, src, tests = gen(rng)
            if st in seen:
                continue
            err = run_tests(src, tests, work)
            if err:
                sys.exit(f"{fam}: {st}\n{err}\n{src}")
            seen.add(st)
            rows.append({"id": f"g{len(rows):04d}", "family": fam, "statement": st, "solution": src.strip()})
            n += 1
    rng.shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} examples written to {a.out}, every solution assembled and tested")


if __name__ == "__main__":
    main()
