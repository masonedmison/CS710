%HW3 Question 1A

%suspects
Suspect(Anne)
Suspect(Betty)
Suspect(Chloe)

%exactly one person is guilty 

% guilty person may lie or tell the truth

TellsTruth(X) :- Innocent(X) .
InTown(X) :- Innocent(X) .
Innocent(X) :- OutOfTown(X) .

Innocent(Betty) :- Innocent(Anne) .
OutOfTown(Betty) :- Innocent(Betty) .
InTown(Betty) :- Innocent(Chloe) .

