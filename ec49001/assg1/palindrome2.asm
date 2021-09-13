MOV 30H,#1H; Load data into RAM as necessary
MOV 31H,#2H
MOV 32H,#3H
MOV 33H,#4H
MOV 34H,#5H
MOV 35H,#6H
MOV 36H,#7H
MOV 37H,#8H
MOV 38H,#8H
MOV 39H,#7H
MOV 3AH,#6H
MOV 3BH,#5H
MOV 3CH,#4H
MOV 3DH,#3H
MOV 3EH,#2H
MOV 3FH,#1H

MOV R1,#30H; Left pointer
MOV R0,#3FH; Right pointer
MOV A,#0; Store 0 to Accumulator
MOV R2,#8; Store the amount of traversing needed
; half the length of sequence
MOV 40H,#01; Default is 0, change to FF if parity breaks

CHECK:; loop for checking parity

MOV A,@R1; Move @R1 to A
MOV B,@R0; Move @R0 to B
SUBB A,B; Subtract A and B
JNZ NEXT; If the above is nonzero, go to NEXT
INC R1; Increase R1 for the next iteration
DEC R0; Decrease R0 for the next iteration

DJNZ R2,CHECK; Perform CHECK R2 times
MOV 40H,#01H; If performed CHECK R2 times, move 01 to 40H
JMP EXIT; Jump to the EXIT

NEXT: MOV 40H,#0FFH; A and B found unequal, change 40H to FF and exit
EXIT:; End the program
END
