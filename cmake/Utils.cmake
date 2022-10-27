function(env_or_default variable_name default_value)
    if(DEFINED ENV{${variable_name}})
        set(${variable_name} "$ENV{${variable_name}}" PARENT_SCOPE)
    else()
        set(${variable_name} "${default_value}" PARENT_SCOPE)
    endif()
endfunction()
