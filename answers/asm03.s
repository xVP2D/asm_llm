section .text
    global main

main:
    mov rax, 0
    call read
    if rax == 0
        mov rax, 1
        print "1337\n"
    else
        mov rax, 2
        print ""
    exit
