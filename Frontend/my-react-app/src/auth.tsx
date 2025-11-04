import React, { useState } from 'react';
import axios from 'axios';
import { motion } from 'motion/react';
import { useNavigate } from 'react-router-dom';
import { API_ENDPOINTS } from './config';

function Login() {
  const navigator = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const post = async () => {
    setError('');
    const data = { username, password };
  
    try {
      const response = await axios.post(API_ENDPOINTS.LOGIN, data, {
        headers: { 'Content-Type': 'application/json' }
      });
  
      if (response.data.access && response.data.refresh) {
        // Save tokens
        localStorage.setItem('access_token', response.data.access);
        localStorage.setItem('refresh_token', response.data.refresh);
        localStorage.setItem('user', JSON.stringify(response.data.user));
  
        // Set default Authorization header
        axios.defaults.headers.common['Authorization'] = `Bearer ${response.data.access}`;
  
        navigator('/dash'); // redirect
      } else {
        setError('Invalid response from server');
      }
    } catch (error) {
      console.error('Error logging in:', error.response?.data || error.message);
      setError(error.response?.data?.detail || 'Login failed. Please try again.');
    }
  };

  return (
    <div>
      <h2>Login</h2>
      <input
        type="text"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        placeholder="Username"
      />
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Password"
      />
      <button onClick={post}>Login</button>
      {error && <p>{error}</p>}
    </div>
  );
}

export default Login;
  