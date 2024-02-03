import React from 'react';
import './box.css';

/**
 * 包装每个组件，统一样式
 * @param props
 * @returns
 */
const Box = (props) => {
  return (
    <div
      style={{ width: '100%', height: '100%' }}
      className='boxWrapper'>
      <div className='boxTitle'>
        {props.title}
      </div>
      <div className='boxContent'>{props.component}</div>
    </div>
  );
};

export default Box;
