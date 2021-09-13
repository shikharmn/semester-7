MOV DPTR, #400H; DPTR at 400H in ROM
MOV R0, #30H; 30H stored in R0
MOV R1, #10H; 16 stored in R1
BACK:
	MOVC A, @A+DPTR; Move ROM data to Accumulator
	INC DPTR; Next data in ROM, increment DPTR
	MOV @R0, A; Move data in A to location stored in R0
	INC R0; Next memory location in R0
	CLR A; Clear A
	DJNZ R1, BACK; Repeat this loop 16 times

ORG 400H
DB 0,1,1,1,0,1,0,0,0,1,0,0,0,1,1,1

COUNT:
MOV A, #00
LOOP:
	MOV P1,A
	INC A
	CJNE A,#0FH, LOOP
MOV P1,A
SJMP COUNT

END