def plot_measured_naive_predicted_1year(y_true, y_naive, y_pred_rf, building_number):
    
    ''' Function to plot 'past' observed data and 'future' predicted and observed data
    Inputs: 
        y_true - observed hourly energy use
        y_naive - prediction from naive model
        y_pred_rf - prediction from random forest model
    '''
    # TODO add subtitle with r2_score_valid, rmsle_valid, mae_valid for each plot type
    
    
    plt.figure(figsize=(20, 10))

    hours_year = np.arange(0,8757,1)
    hours_3months = np.arange(0,2160,1) 
    hours_next9months = np.arange(2161, 8757, 1) 

    # For whole year
    plt.plot(hours_3months, y_true[building_number][:2160], c = 'navy', label = "3 months known data", linewidth = 0.5);
    plt.title('3 months of known data and 9 months of predictions',  fontsize = 'xx-large')

    plt.xlabel('hour of year', fontsize = 'xx-large')
    plt.ylabel('Electricity consumption in kWh', fontsize = 'xx-large')
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

    plt.plot(hours_next9months, y_naive[building_number][2161:8757], c = 'goldenrod', label = "Predicted - naive", linewidth = 0.5); 

    plt.plot(hours_next9months, y_pred_rf[building_number][2161:8757], c = 'seagreen', label = "Predicted - RF", linewidth = 0.5); 

    plt.plot(hours_next9months, y_true[building_number][2161:8757], c = 'dodgerblue', label = "Actual", linewidth = 0.5);

    plt.legend( loc = "upper left", fontsize = 'x-large');

    plt.show()
    
    return
