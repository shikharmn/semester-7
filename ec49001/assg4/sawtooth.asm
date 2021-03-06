START: ; Sawtooth
CLR P0.7
MOV R1, #05H

; Storing values

MOV 5, #0
MOV 6, #39
MOV 7, #72
MOV 8, #100
MOV 9, #124
MOV 10, #144
MOV 11, #161
MOV 12, #175
MOV 13, #187
MOV 14, #198
MOV 15, #206
MOV 16, #214
MOV 17, #220
MOV 18, #225
MOV 19, #230
MOV 20, #234
MOV 21, #237
MOV 22, #255
MOV 23, #215
MOV 24, #182
MOV 25, #154
MOV 26, #130
MOV 27, #110
MOV 28, #93
MOV 29, #79
MOV 30, #67
MOV 31, #56
MOV 32, #48
MOV 33, #40
MOV 34, #34
MOV 35, #29
MOV 36, #24

LOOP:
MOV A, @R1			; Stores value of function from @R1
MOV P1, A
INC R1

CJNE R1,#37,LOOP	; If not reached the end of memory, keep looping
MOV R1, #05H		; Start again
SJMP LOOP

DELAY:		; Delay module
MOV R0,#100
DJNZ R0,$
RET