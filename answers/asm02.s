section .text
    global main

main:
    read from stdin
    if (input == '42') then
        print '1337'
        exit 0
    else
        print ''
        exit 1
