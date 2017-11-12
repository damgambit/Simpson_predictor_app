/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 * @flow
 */

import React, { Component } from 'react';
import {
  Platform,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image
} from 'react-native';


var axios = require('axios')

var ImagePicker = require('react-native-image-picker');

export default class App extends Component<{}> {
  constructor(props) {
    super(props);
    this.state = { 
      image: null,
      prediction: null
    };
  };
  
  onclick() {
    var options = {
      title: 'Select Simpson Character Image',
      storageOptions: {
        skipBackup: true,
        path: 'images'
      }
    };


    ImagePicker.showImagePicker(options, (response) => {

      if (response.didCancel) {
        console.log('User cancelled image picker');
      }
      else if (response.error) {
        console.log('ImagePicker Error: ', response.error);
      }
      else if (response.customButton) {
        console.log('User tapped custom button: ', response.customButton);
      }
      else {
        let source = { uri: response.uri };

        // You can also display the image using data:
        // let source = { uri: 'data:image/jpeg;base64,' + response.data };

        this.setState({
          image: response.data
        });
      }
    });
    
  };

  renderImage() {
    if(this.state.image !== null) {
      return (
        <View style={styles.img}>
          <Image style={{ width: 320, height: 320 }} source={{ uri: `data:image/jpg;base64,${this.state.image}`}} />
        </View>
      )
    }
  };

  renderPrediction() {
    if(this.state.prediction !== null) {
      return (
        <View style={styles.pred}>
          <Text style={styles.predText}>
            {this.state.prediction}
          </Text>
        </View>
      )
    }
  }

  predict() {
    if(this.state.image == null) {
      this.setState({
        prediction: 'Please pick an Image.'
      })
    } else {
      // send this.state.image to the flask api
      console.log('sending')
      axios({ url: 'http://192.168.1.8:5000/api/predict', method: 'POST', 
              data: { image: this.state.image } }).then((response) => {
        
        this.setState({
          prediction: response.data.prediction
        })

      }).catch((error) => {
        console.log(error)
        this.setState({
          prediction: "Could not predict the character! Please try again."
        })
      })

      // take the response and pass it to this.state.prediction
    }
  }

  render() {
    return (
      <View style={styles.container}>
        {this.renderPrediction()}

        {this.renderImage()}

        <View style={styles.buttons}>
          <TouchableOpacity onPress={this.onclick.bind(this)}>
            <Text style={styles.buttonText}>
              Pick an Image
            </Text>
          </TouchableOpacity>

          <TouchableOpacity onPress={this.predict.bind(this)}>
            <Text style={styles.buttonText}>
              Predict Simpson Character
            </Text>
          </TouchableOpacity>
        </View>
        
        <Image source={this.state.avatarSource} style={styles.uploadAvatar} />
        
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'space-around',
    backgroundColor: 'yellow',
    flexDirection: 'column'
  },
  img: {
    flex: 1,
    marginTop: 35,
    alignItems: 'center',
    justifyContent: 'center'
  },
  buttons: {
    flex: 1,
    marginTop: 10,
    alignItems: 'center',
    justifyContent: 'center'
  },
  buttonText: {
    padding: 10,
    fontSize: 22,
    color: 'grey'
  },
  pred: {
    flex: 1,
    marginTop: 15,
    alignItems: 'center',
    justifyContent: 'center'
  },
  predText: {
    color: 'red',
    fontSize: 28
  }
});
