import sys
import logging

def error_message_detail(error: Exception, error_detail: sys)->str:
    """This function generates error message with details."""

    _,_,exc_tab= error_detail.exc_info()

    file_name= exc_tab.tb_frame.f_code.co_filename

    line_number= exc_tab.tb_lineno
    error_message= f"Error occured in python script :[{file_name}] at line_number: [{line_number}] : {str(error)}"

    logging.error(error_message)

    return error_message

class MyException(Exception):
    """This is a custom exception class."""

    def __init__(self, error_message:str, error_detail:sys):
        super().__init__(error_message)
        print( " The error_message is :",error_message)
        self.error_message= error_message_detail(error_message, error_detail)

    def __str__(self) ->str:
        return self.error_message