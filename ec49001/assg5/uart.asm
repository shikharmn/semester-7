ORG 0000H
CLR SM0
SETB SM1; Serial mode in 8-bit UART mode
SETB REN; Serial port receiver enabled
       
MOV PCON, #128

MOV TMOD, #20H
MOV TH1, #243; Calculation for this
MOV TL1, #243; Explained in report
SETB TR1
MOV R1, #30H; This is where storage begins
MOV R2, #9; Size of roll number
       
LOOP:; Loop for receiving roll number input
JNB RI, $; Wait for data to be received
CLR RI
MOV A, SBUF; Move input in buffer to ACC
MOV @R1, A; Store input to memory
INC R1; Increment memory pointer
DJNZ R2, LOOP; Loop for the size of roll number
       
MOV R2, #9; Set size variable
MOV R1, #30H; and memory pointer again
       
LOOP2:; Loop for transmitting stored roll number
MOV SBUF, @R1; Move stored data to buffer
INC R1; Increment memory pointer
WAIT: JNB TI, WAIT; Wait for data to be transmitted
CLR TI
DJNZ R2, LOOP2; Loop for the size of roll number
