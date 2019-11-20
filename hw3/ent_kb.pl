%Homework 3 Question 2A

%Definitions 

%departments 
department(Production)
department(Research)
department(Administration)
department(Trade)
department(HumanResources)
department(PublicRelations)
%--------------------------

employee(X) :- is-employed-by(X, enterprise); is-employed-by(X, is-part-of(enterprise) .

admin-employee(X) :-is-employed-by(X, department(Administration); is-employed-by(X, is-part-of(department(Administration) . 

hitech-ent(X) :- has-part(X, research-department) .

%revisit 
indust-ent(X) :- has-part(X, prod-department), employs(X) ...


%Facts (assertions) 
big-ent(Alcatel) .

has-part(Alcatel, [AD1, RD1, HRD1])) .

office(OFF1) .
is-part-of(OFF1, RD1) .

office(OFF2) .
is-part-of(OFF2, RD1) .

office(OFF3) .
is-part-of(OFF3, AD1) .

is-employed-by(Joe, OFF3) .

is-employed-by(Anne, OFF3) .

manages(Jim, AD3) .

manages(Bob, OFF3) .

manages(Jim, Alcatel) .

fam-based-ent(SmithBrothers) .

is-employed-by(Frank, SmithBrothers) .

is-employed-by(Lea, SmithBrothers) .

is-employed-by(Dave, SmithBrothers) .

is-employed-by(Kate, SmithBrothers) .

is-employed-by(Dino, SmithBrothers) .

