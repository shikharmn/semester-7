MOV 30H,#247; Store the number in 247
MOV A,30H; Move the number to accumulator
MOV B,#10; Store 10 in B
DIV AB; Divide A by 10
MOV R0,B; Store the remainder in R0
MOV B,#10; Store 10 in B
DIV AB; Divide A by 10 again
MOV 40H,A; Store the 100s digit in A
MOV A,B; Shift the 10s digit to A
MOV B,#16; Set B to 16
MUL AB; Multiply 16 to the 10s digit
ADD A,R0; Add the initial remainder to A
MOV 41H,A; Move this to 41H
END
