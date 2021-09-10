START:
MOV 30H,#16H
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

MOV R3,#0FH

OUT_LOOP:
MOV B,R3
MOV R4,B
MOV R0,#30H

IN_LOOP:
MOV B,@R0
INC R0
MOV A,@R0
CJNE A,B,NEXT
NEXT: JNC NO_SWP
MOV @R0,B
DEC R0
MOV @R0,A
INC R0

NO_SWP:
DJNZ R4,IN_LOOP
DJNZ R3,OUT_LOOP
END
