###################################################################################
# MIT License

# Copyright (c) 2015-2017 Gregory Paradis

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###################################################################################

import pulp as plp

SENSE_MINIMIZE = plp.LpMinimize
SENSE_MAXIMIZE = plp.LpMaximize
SENSE_EQ = plp.LpConstraintEQ
SENSE_GEQ = plp.LpConstraintGE
SENSE_LEQ = plp.LpConstraintLE
VTYPE_INTEGER = 'Integer'
VTYPE_BINARY = 'Binary'
VTYPE_CONTINUOUS = 'Continuous'
VBNDS_INF = float('inf')


class Variable:
    def __init__(self, name, vtype, lb=0., ub=VBNDS_INF, val=None):
        self.name = name
        self.vtype = vtype
        self.lb = lb
        self.ub = ub
        self.val = val
        if vtype == VTYPE_CONTINUOUS:
            self.var = plp.LpVariable(name, lowBound=lb, upBound=ub, cat=plp.LpContinuous)
        elif vtype == VTYPE_INTEGER:
            self.var = plp.LpVariable(name, lowBound=lb, upBound=ub, cat=plp.LpInteger)
        elif vtype == VTYPE_BINARY:
            self.var = plp.LpVariable(name, cat=plp.LpBinary)


class Constraint:
    """
    Encapsulates data describing a constraint in an optimization problem. This includes a constraint name (should be unique within a problem, although the user is responsible for enforcing this condition), a vector of coefficient values (length of vector should match the number of variables in the problem, although the user is responsible for enforcing this condition), a sense (should be one of ``SENSE_EQ``, ``SENSE_GEQ``, or ``SENSE_LEQ``), and a right-hand-side value.
    """
    def __init__(self, name, coeffs, sense, rhs):
        self.name = name
        self.coeffs = coeffs
        self.sense = sense
        self.rhs = rhs


class Problem:
    def __init__(self, name, sense=SENSE_MAXIMIZE):
        self._name = name
        self._vars = {}
        self._z = {}
        self._constraints = {}
        self._solution = None
        self._sense = sense
        self._prob = plp.LpProblem(name, sense)

    def add_var(self, name, vtype, lb=0., ub=VBNDS_INF):
        self._vars[name] = Variable(name, vtype, lb, ub)
        self._solution = None

    def var_names(self):
        return list(self._vars.keys())

    def constraint_names(self):
        return list(self._constraints.keys())

    def name(self):
        return self._name
        
    def var(self, name):
        return self._vars[name]

    def sense(self, val=None):
        if val:
            self._sense = val
            self._prob.sense = val
            self._solution = None
        else:
            return self._sense

    def solved(self):
        return self._solution is not None
        
    def z(self, coeffs=None, validate=False):
        if coeffs:
            if validate:
                for v in coeffs:
                    assert v in self._vars
            self._z = coeffs
            self._prob += plp.lpSum([coeffs[v] * self._vars[v].var for v in coeffs]), "Objective"
            self._solution = None
        else:
            assert self.solved()
            return plp.value(self._prob.objective)
        
    def add_constraint(self, name, coeffs, sense, rhs, validate=False):
        if validate:
            for v in coeffs:
                assert v in self._vars
        constraint_expr = plp.lpSum([coeffs[v] * self._vars[v].var for v in coeffs])
        if sense == SENSE_EQ:
            self._prob += (constraint_expr == rhs), name
        elif sense == SENSE_GEQ:
            self._prob += (constraint_expr >= rhs), name
        elif sense == SENSE_LEQ:
            self._prob += (constraint_expr <= rhs), name
        self._constraints[name] = Constraint(name, coeffs, sense, rhs)
        self._solution = None

    def solution(self):
        return {x: self._vars[x].var.varValue for x in self._vars}
    
    def solve(self):
        self._prob.solve()
        if plp.LpStatus[self._prob.status] == 'Optimal':
            self._solution = {v.name: v.var.varValue for v in self._vars.values()}
        return self._prob


def _solve_pulp(self):
    import pulp
    # Initialize PuLP model
    self._m = m = pulp.LpProblem(self._name, pulp.LpMaximize if self._sense == SENSE_MAXIMIZE else pulp.LpMinimize)

    # Initialize variables
    vars = {v.name: pulp.LpVariable(name=v.name, lowBound=0, cat=pulp.LpContinuous) for v in self._vars.values()}

    # Objective function
    z = pulp.LpAffineExpression([(vars[v], self._z[v]) for v in vars])
    m += z  # Objective function setup

    # Constraints
    for name, constraint in self._constraints.items():
        lhs = pulp.LpAffineExpression([(vars[x], constraint.coeffs[x]) for x in constraint.coeffs])
        if constraint.sense == SENSE_EQ:
            m += lhs == constraint.rhs
        elif constraint.sense == SENSE_LEQ:
            m += lhs <= constraint.rhs
        elif constraint.sense == SENSE_GEQ:
            m += lhs >= constraint.rhs

    # Solve the problem
    m.solve()

    # Handle results
    if pulp.LpStatus[m.status] == 'Optimal':
        for k, v in self._vars.items():
            v._solver_var = vars[k]
            v.val = vars[k].varValue

    return m

