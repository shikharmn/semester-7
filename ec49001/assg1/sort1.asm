START:
MOV 30H,#16H; Load data as required
MOV 31H,#7H
MOV 32H,#15H 
MOV 33H,#2H
MOV 34H,#10H 
MOV 35H,#11H
MOV 36H,#12H
MOV 37H,#9H
MOV 38H,#5H
MOV 39H,#1H
MOV 3AH,#8H
MOV 3BH,#6H
MOV 3CH,#3H
MOV 3DH,#14H 
MOV 3EH,#4H
MOV 3FH,#13H

MOV R3,#0FH; Size of our array

OUT_LOOP:; Control for the outer loop
MOV B,R3
MOV R4,B; Copy the value in R3 to R4
MOV R0,#30H; Initialise the iterator

IN_LOOP:
MOV B,@R0; Store @R0 to B
INC R0; Increment R0
MOV A,@R0; Store @R0 to A
CJNE A,B,NEXT; Carry is 1 if A>B
NEXT: JNC NO_SWP; If carry is 0, jump to NO_SWP and skip the swap
MOV @R0,B; Following lines swap the two elements
DEC R0
MOV @R0,A
INC R0

NO_SWP:; Skip swap if not needed
DJNZ R4,IN_LOOP; Perform the comparison N times, making a single pass
DJNZ R3,OUT_LOOP; Perform the pass N times
END
