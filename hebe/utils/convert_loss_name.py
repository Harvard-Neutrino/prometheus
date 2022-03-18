# Conversion of energy loss names
def convert_loss_name(type_of_loss):
    '''
    Converts from PROPOSAL naming convention of energy losses to PPC ones.
    '''
    if type_of_loss == 'epair':
        return 'epair'
    elif type_of_loss == 'brems':
        return 'brems'
    elif type_of_loss == 'photo' or type_of_loss == 'hadr':
        return 'hadr'
    elif type_of_loss == 'ioniz':
        return 'delta'
    # Continuous loss is the same as ionization; just named differently in PROPOSAL
    elif type_of_loss == 'continuous':
        return 'delta'
    else:
        raise Exception('Invalid energy loss', type_of_loss)
