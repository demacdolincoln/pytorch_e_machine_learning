from torch import nn

class Ex1(nn.Module):
    '''poucas camadas, hidden size baixo'''
    def __init__(self, input_size, hidden_size, n_classes):
        super(Ex1, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x, dropout):
        if dropout == True:
            x = nn.functional.dropout(x)
        x = self.l1(x)
        x = self.l2(x)
        return x

class Ex2(nn.Module):
    '''poucas camadas, hidden size alto'''
    def __init__(self, input_size, hidden_size, n_classes):
        super(Ex2, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x, dropout):
        if dropout == True:
            x = nn.functional.dropout(x)
        x = self.l1(x)
        x = self.l2(x)
        return x

class Ex3(nn.Module):
    '''muitas camadas, hidden size baixo'''
    def __init__(self, input_size, hidden_size, n_classes):
        super(Ex3, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, hidden_size)
        self.l7 = nn.Linear(hidden_size, n_classes)

    def forward(self, x, dropout):
        if dropout == True:
            x = nn.functional.dropout(x)

        for i in range(1, 8):
            x = getattr('l{i}')(x)
        return x


class Ex4(nn.Module):
    '''muitas camadas, hidden size alto'''
    def __init__(self, input_size, hidden_size, n_classes):
        super(Ex4, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, hidden_size)
        self.l7 = nn.Linear(hidden_size, n_classes)

    def forward(self, x, dropout):
        if dropout == True:
            x = nn.functional.dropout(x)

        for i in range(1, 8):
            x = getattr('l{i}')(x)
        return x
